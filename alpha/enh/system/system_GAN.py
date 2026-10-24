# GAN based speech enhancement
import random
import torch
import functools
import numpy as np
from omegaconf import OmegaConf
from copy import deepcopy
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR_scheduler
from modules.utils.logging import logger
from modules.blocks import discriminator
from modules.utils.common import EPS, refine_state_dict, resolve_path, freeze_model
from .system import BaseSE, ComplexSE, UniSE


class GAN_Base(BaseSE):
    def __init__(self, conf):
        super().__init__(conf)
        self.automatic_optimization = False
        self.d_model = getattr(discriminator, conf['d_model']['name'])(conf['d_model'])
        self.train_padding = conf['system'].get('train_padding', False)
        self.n_critic = conf['d_model'].get('n_critic', 1)
        self.SE_loss_w = conf['loss'].get('weight', 1.0)
        self.gp_w = conf.get('gp_w', 0.0) # gradient penalty weight
        self.Ra = conf.get('Ra', False) # Relativistic average
        self.self2self = conf.get('self2self', conf.get('random_output', False)) # self to self learning        
        self.teacher = conf.get('teacher', None)
        if self.teacher is not None:
            self.teacher_loss = self.get_loss(self.teacher['loss'])
            self.teacher_mask_loss = self.get_loss(self.teacher['mask_loss'])
        self.muti_branch = conf.get('muti_branch', False)
        self.final_SE_loss = conf.get('final_SE_loss', False)
        self.clip_grad = functools.partial(self.clip_gradients, 
                                           gradient_clip_val=conf['system']['gradient_clip_val'], 
                                           gradient_clip_algorithm="norm")
    
    def get_teacher_loss(self, noisy_wav, est_wav, teacher, spec_mask, mag_mask):
        loss = 0.0
        if self.teacher['loss']['weight'] > 0:
            loss = loss + self.teacher['loss']['weight'] * self.teacher_loss(teacher, est_wav)
        if self.teacher['mask_loss']['weight'] > 0:
            noisy_mag = torch.abs(self.stft.apply_stft(noisy_wav))
            teacher_mag = torch.abs(self.stft.apply_stft(teacher))
            est_noise_mag = noisy_mag - teacher_mag
            if self.teacher['mask'] == "IBM":
                teacher_mask = (teacher_mag > est_noise_mag).float()
            elif self.teacher['mask'] == "IRM":
                teacher_mask = teacher_mag / (teacher_mag + est_noise_mag).clip(EPS)
            elif self.teacher['mask'] == "WFM":
                teacher_mask = teacher_mag**2 / (teacher_mag**2 + est_noise_mag**2).clip(EPS)
            else:
                raise NotImplementedError
            if mag_mask is None:
                mag_mask = torch.abs(spec_mask)
            loss = loss + self.teacher['mask_loss']['weight'] * self.teacher_mask_loss(
                teacher_mask.squeeze(1), mag_mask.squeeze(1)) 
        return loss

    def d_loss(self, disc_real_outputs, disc_generated_outputs):
        raise NotImplementedError

    def g_loss(self, disc_real_outputs, disc_generated_outputs):
        raise NotImplementedError

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _, _, _ = self.d_model(interpolates, fake_samples)
        gp_loss = 0
        for d_interpolate in d_interpolates:
            fake = torch.ones_like(d_interpolate)
            gradients = torch.autograd.grad(outputs=d_interpolate, inputs=interpolates,
                grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1).to(self.device)
            gp_loss += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp_loss    

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.teacher:
            clean_wav, noisy_wav, teacher, _ = batch
        else:
            clean_wav, noisy_wav, _ = batch
        # train discriminator
        if optimizer_idx == 0:
            _, est_wav = self.forward(noisy_wav, clean_wav)
            est_wav, clean_wav = est_wav.unsqueeze(1), clean_wav.unsqueeze(1)
            y_df_hat_r, y_df_hat_g, _, _ = self.d_model(clean_wav, est_wav.detach())
            d_loss = self.d_loss(y_df_hat_r, y_df_hat_g)
            self.log("d/d_loss", d_loss)
            if self.gp_w:
                loss_gp = self.gp_w * self.compute_gradient_penalty(clean_wav.data, est_wav.data)
                self.log("d/loss_gp", loss_gp)
                d_loss = loss_gp + d_loss
            self.log("d/loss", d_loss) # the final discriminator loss
            return d_loss
        # train generator
        if optimizer_idx == 1:
            target_wav = noisy_wav if self.self2self else clean_wav
            if self.teacher:
                SE_loss, est_wav, spec_mask, mag_mask = self.forward(noisy_wav, target_wav, ret_mask=True)
            else:
                SE_loss, est_wav = self.forward(noisy_wav, target_wav)
            if self.muti_branch:
                est_wav = self.muti_branch * teacher + (1 - self.muti_branch) * est_wav
                if self.final_SE_loss is True:
                    SE_loss = self.loss_fun(clean_wav, est_wav)
            y_df_hat_r, y_df_hat_g, _, _ = self.d_model(clean_wav.unsqueeze(1), 
                                                        est_wav.unsqueeze(1))
            g_loss = self.g_loss(y_df_hat_r, y_df_hat_g)
            self.log("g/g_loss", g_loss)
            if self.SE_loss_w is not None:
                SE_loss = self.SE_loss_w * SE_loss
                self.log("g/SE_loss", SE_loss)
                g_loss = g_loss + SE_loss
            if self.muti_branch == False:
                if self.teacher:
                    teacher_loss = self.get_teacher_loss(noisy_wav, est_wav, teacher, spec_mask, mag_mask)
                    self.log("g/teacher_loss", teacher_loss)
                    g_loss = g_loss + teacher_loss
            self.log("g/loss", g_loss) # the final generator loss
            return g_loss

    def get_optimizer(self, opt_conf, parameters):
        # overload the BaseModule
        _opt_conf = deepcopy(opt_conf)
        name = _opt_conf.pop('name', 'Adam')
        lr_scheduler_conf = _opt_conf.pop('lr_scheduler', {})
        optimizer = getattr(torch.optim, name)(parameters, **_opt_conf)
        if lr_scheduler_conf:
            name = lr_scheduler_conf.pop('name', 'ExponentialLR')
            if name == 'ExponentialLR':
                last_epoch = self.current_epoch if self.current_epoch > 0 else -1
                logger.debug(f'last_epoch: {last_epoch}')
                lr_scheduler_conf['last_epoch'] = last_epoch
            scheduler = getattr(LR_scheduler, name)(optimizer, **lr_scheduler_conf)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer
        
    def configure_optimizers(self):
        opt_conf = OmegaConf.to_container(self.conf['optimizer'])
        return (self.get_optimizer(opt_conf['generator'], self.model.parameters()),
                self.get_optimizer(opt_conf['discriminator'], self.d_model.parameters()))
        

class GAN(GAN_Base):
    # LSGAN
    def __init__(self, conf):
        super().__init__(conf)
    
    def d_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            if self.Ra is True:
                r_loss = torch.mean((dr - torch.mean(dg) - 1)**2)
                g_loss = torch.mean((dg - torch.mean(dr) + 1)**2)
            else:
                r_loss = torch.mean((1 - dr)**2)
                g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
        return loss
    
    def g_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            if self.Ra is True:                
                l = torch.mean((dg - torch.mean(dr) - 1)**2) + \
                    torch.mean((dr - torch.mean(dg) + 1)**2)
            else:
                l = torch.mean((1 - dg)**2)
            loss += l
        return loss    


class ComplexSE_GAN(GAN, ComplexSE):
    # Speech Enhancement ComplexSE and GAN
    def __init__(self, conf):
        super().__init__(conf)

SE_GAN = ComplexSE_GAN


class UniSE_GAN(GAN, UniSE):
    # Speech Enhancement UniSE and GAN
    def __init__(self, conf):
        super().__init__(conf)


class WGAN(GAN_Base):
    # WGAN
    def __init__(self, conf):
        super().__init__(conf)
        self.gp_w = conf.get('gp_w', 10) # gradient penalty weight
    
    def d_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):            
            if self.Ra is True:
                r_loss = torch.mean(dr - torch.mean(dg))
                g_loss = torch.mean(dg - torch.mean(dr))
            else:
                r_loss = torch.mean(dr)
                g_loss = torch.mean(dg)
            loss += (-r_loss + g_loss)
        return loss

    def g_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            if self.Ra is True:
                l = - torch.mean(dg - torch.mean(dr)) + torch.mean(dr - torch.mean(dg))
            else:
                l = - torch.mean(dg)
            loss += l
        return loss    


class ComplexSE_WGAN(WGAN, ComplexSE):
    # Speech Enhancement with ComplexSE and WGAN-GP
    def __init__(self, conf):
        super().__init__(conf)

UN_SE_WGAN = ComplexSE_WGAN


class UniSE_WGAN(WGAN, UniSE):
    # Speech Enhancement with UniSE and WGAN-GP
    def __init__(self, conf):
        conf['mag_data'] = conf.get('mag_data', False)
        conf['bidirectional'] = conf.get('bidirectional', True)
        conf['mag_decoder'] = conf.get('mag_decoder', False)
        conf['spec_decoder'] = conf.get('spec_decoder', True)
        conf['norm'] = conf.get('norm', 'batch')
        super().__init__(conf)


class MagSE_WGAN(WGAN, UniSE):
    # Speech Enhancement with UniSE (magnitude only) and WGAN-GP
    def __init__(self, conf):
        conf['mag_data'] = conf.get('mag_data', True) # mag only
        conf['spec_data'] = conf.get('spec_data', False)
        conf['bidirectional'] = conf.get('bidirectional', True)
        conf['mag_decoder'] = conf.get('mag_decoder', True) # mag only
        conf['spec_decoder'] = conf.get('spec_decoder', False)
        conf['norm'] = conf.get('norm', 'batch')
        # loss
        conf['loss_fun_w'] = conf.get('loss_fun_w', 0) # time-domain loss
        conf['mag_loss'] = conf.get('mag_loss', 1.0) # magnitude MSE loss
        
        super().__init__(conf)


class UnSE_MOS_GAN(GAN_Base, UniSE):
    # Unsupervised speech enhancement using mean opinion score (MOS)
    def __init__(self, conf):
        super().__init__(conf)        
        self.MOS_norm = conf['DNSMOS'].get('norm', True)
        self.MOS_random = conf['DNSMOS'].get('random', False)
        self.est_score = conf['d_model'].get('est_score', None)
        self.noise_score = conf['d_model'].get('noise_score', None)
        
    def process_mos(self, mos):
        if self.MOS_norm:
            mos = mos / 5.0
        if isinstance(self.MOS_random, float):
            mos = mos + self.MOS_random * ((torch.rand_like(mos) - 0.5) * 2)
            mos = torch.clamp(mos, min=0.2, max=1.0)
        return mos.unsqueeze(-1)
    
    def training_step(self, batch, batch_idx):
        noisy_wav = batch['noisy_wav']
        noisy_metric = self.process_mos(batch['label'].data)
        if self.train_padding:
            noisy_wav = self.padding(noisy_wav)
        torch_one = torch.ones(noisy_wav.shape[0], self.conf['d_model']['out_size'], 
                               device=noisy_wav.device).float()
        torch_zero = torch.zeros_like(torch_one)
        g_opt, d_opt = self.optimizers()

        # train discriminator
        est_wav = self.forward(noisy_wav, train=False).detach()
        noisy_score = self.d_model(torch.abs(self.stft.apply_stft(noisy_wav)))
        d_loss = F.mse_loss(noisy_metric, noisy_score)
        if self.est_score is not None:
            est_score = self.d_model(torch.abs(self.stft.apply_stft(est_wav)))
            d_loss = d_loss + F.mse_loss(est_score, torch_one)
            self.log("d/est_score", est_score.mean())
        if self.noise_score is not None:
            est_noise = noisy_wav - est_wav
            noise_score = self.d_model(torch.abs(self.stft.apply_stft(est_noise)))
            d_loss = d_loss + F.mse_loss(noise_score, torch_zero)
            self.log("d/noise_score", noise_score.mean())
        self.log("d/loss", d_loss)
        self.log("d/noisy_score", noisy_score.mean())
        self.toggle_optimizer(d_opt)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        self.clip_grad(d_opt)
        d_opt.step()
        self.untoggle_optimizer(d_opt)
    
        # Optimize SE model
        if self.global_step % self.n_critic == 0:
            if self.teacher is not None:
                SE_loss, est_wav, spec_mask, mag_mask = self.forward(noisy_wav, noisy_wav, ret_mask=True)
            else:
                SE_loss, est_wav = self.forward(noisy_wav, noisy_wav)
            est_score = self.d_model(torch.abs(self.stft.apply_stft(est_wav)))
            g_loss = F.mse_loss(est_score, torch_one)
            if self.SE_loss_w is not None:
                SE_loss = self.SE_loss_w * SE_loss
                self.log("g/SE_loss", SE_loss)
                self.log("g/g_loss", g_loss)
                g_loss = g_loss + SE_loss            
            if self.teacher is not None:
                teacher_wav = batch['teacher_wav']
                if self.train_padding:
                    n_padding = noisy_wav.shape[-1] - teacher_wav.shape[-1]
                    teacher_wav = F.pad(teacher_wav, (0, n_padding))
                teacher_loss = self.get_teacher_loss(noisy_wav, est_wav, teacher_wav, spec_mask, mag_mask)
                self.log("g/teacher_loss", teacher_loss)
                g_loss = g_loss + teacher_loss
            self.log("g/loss", g_loss)
            self.toggle_optimizer(g_opt)
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.clip_grad(g_opt)
            g_opt.step()
            self.untoggle_optimizer(g_opt)
            

class UnSE_TTA(GAN_Base, UniSE):
    # Test Time Adaptation
    def __init__(self, conf):
        super().__init__(conf)
        self.UbyU = conf['system'].get('UbyU') # utterance by utterance
        self.n_steps = conf['system'].get('n_steps', 1)
        
    def init_model(self, model_conf):
        model_path = model_conf.get('init')
        assert model_path is not None
        model_path = resolve_path(model_path)
        logger.info(f'Model initialize with: {model_path}')
        state_dict = torch.load(model_path)['state_dict']
        self.model.load_state_dict(refine_state_dict(state_dict, prefix='model.'))
        
        self.d_model.load_state_dict(refine_state_dict(state_dict, prefix='d_model.'))
        self.d_model.eval()
        self.d_model = freeze_model(self.d_model) # freeze the model !
        self.triple = False
        names = set([item.split('.')[0] for item in state_dict.keys()])
        if 'd2_model' in names: # UnSE_TripleGAN
            self.triple = True
            self.d2_model = deepcopy(self.d_model)
            self.d2_model.load_state_dict(refine_state_dict(state_dict, prefix='d2_model.'))
            self.d2_model.eval()
            self.d2_model = freeze_model(self.d2_model) # freeze the model !
            self.d3_model = deepcopy(self.d_model)
            self.d3_model.load_state_dict(refine_state_dict(state_dict, prefix='d3_model.'))
            self.d3_model.eval()
            self.d3_model = freeze_model(self.d3_model) # freeze the model !
            self.d_model_list = [self.d_model, self.d2_model, self.d3_model]
    
    def training_step(self, batch, batch_idx):
        if self.UbyU:
            self.init_model(self.conf['model'])
        clean_wav, noisy_wav = batch['clean_wav'], batch['noisy_wav']
        noisy_wav, _ = self.data_augment(noisy_wav, clean_wav)
        if self.train_padding:
            noisy_wav = self.padding(noisy_wav)
        torch_one = torch.ones(noisy_wav.shape[0], self.conf['d_model']['out_size'], 
                               device=noisy_wav.device).float()        
        g_opt = self.optimizers()[0]
        for _ in range(self.n_steps):
            if self.teacher is not None:
                SE_loss, est_wav, spec_mask, mag_mask = self.forward(noisy_wav, noisy_wav, ret_mask=True)
            else:
                SE_loss, est_wav = self.forward(noisy_wav, noisy_wav)
            est_mag = torch.abs(self.stft.apply_stft(est_wav))
            if self.triple is False:
                g_loss = F.mse_loss(self.d_model(est_mag), torch_one)
            else:
                g_loss = sum(F.mse_loss(d_model(est_mag), torch_one) for d_model in self.d_model_list)
            if self.SE_loss_w is not None:
                SE_loss = self.SE_loss_w * SE_loss
                self.log("g/SE_loss", SE_loss)
                self.log("g/g_loss", g_loss)
                g_loss = g_loss + SE_loss
            if self.teacher is not None:
                teacher_wav = batch['teacher_wav']
                if self.train_padding:
                    n_padding = noisy_wav.shape[-1] - teacher_wav.shape[-1]
                    teacher_wav = F.pad(teacher_wav, (0, n_padding))
                teacher_loss = self.get_teacher_loss(noisy_wav, est_wav, teacher_wav, spec_mask, mag_mask)
                self.log("g/teacher_loss", teacher_loss)
                g_loss = g_loss + teacher_loss
            self.log("g/loss", g_loss)
            self.toggle_optimizer(g_opt)
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.clip_grad(g_opt)
            g_opt.step()
            self.untoggle_optimizer(g_opt)
        if self.UbyU:
            self.save_wav_file(est_wav, clean_wav, noisy_wav, batch['wav_id'])
        

from modules.utils.compute_score import ComputePESQ, ComputeMOS, TorchMOS
class UnSE_MetricGAN(GAN_Base, UniSE):
    metric_dict = {'SIG': 0, 'BAK':1, 'OVL':2, 'ALL': -1}
    def __init__(self, conf):
        super().__init__(conf)
        torch_worker = conf['DNSMOS'].get('torch_worker', False)
        if torch_worker:
            self.score_worker = TorchMOS(conf)
        else:
            self.score_worker = self.MOS_worker if hasattr(
                self, 'MOS_worker') else ComputeMOS(conf)
        self.MOS_norm = conf['DNSMOS'].get('norm', True)
        self.MOS_random = conf['DNSMOS'].get('random', False)
        self.score_idx = self.metric_dict[conf['DNSMOS']['metric']]
        if self.score_idx == -1:
            assert conf['d_model']['out_size'] == 3
        self.noise_score = conf['d_model'].get('noise_score', None)
        self.noise_weight = conf['d_model'].get('noise_weight', 1.0)
        self.score_increment = conf['DNSMOS'].get('score_increment', None)
        
    def metric_score(self, wav, score_idx=-1):
        score = self.score_worker.batch_scores(wav)
        if not isinstance(score, torch.Tensor):
            score = torch.tensor(score).type_as(wav)
        if score_idx != -1:
            score = score[:, score_idx]
        if self.MOS_norm:
            score = score / 5.0
        if score.ndim == 1:
            score = score.unsqueeze(-1)
        if isinstance(self.MOS_random, float):
            score = score + self.MOS_random * ((torch.rand_like(score) - 0.5) * 2)
            score = torch.clamp(score, min=0.2, max=1.0)
        return score
    
    def training_step(self, batch, batch_idx):
        clean_wav, noisy_wav = batch['clean_wav'], batch['noisy_wav']
        noisy_wav, _ = self.data_augment(noisy_wav, clean_wav)
        if self.train_padding:
            noisy_wav = self.padding(noisy_wav)
        torch_one = torch.ones(noisy_wav.shape[0], self.conf['d_model']['out_size'], 
                               device=noisy_wav.device).float()
        g_opt, d_opt = self.optimizers()

        # train discriminator
        est_wav = self.forward(noisy_wav, train=False).detach()
        est_score = self.d_model(torch.abs(self.stft.apply_stft(est_wav)))
        est_metric = self.metric_score(est_wav, self.score_idx)
        noisy_score = self.d_model(torch.abs(self.stft.apply_stft(noisy_wav)))
        noisy_metric = self.metric_score(noisy_wav, self.score_idx)
        d_loss = F.mse_loss(est_metric, est_score) + F.mse_loss(noisy_metric, noisy_score)
        if self.noise_score is not None:
            est_noise = noisy_wav - est_wav
            noise_score = self.d_model(torch.abs(self.stft.apply_stft(est_noise)))
            noise_metric = self.metric_score(est_noise, self.score_idx)
            d_loss = d_loss + F.mse_loss(noise_metric, noise_score)
            self.log("d/noise_score", noise_score.mean())
            self.log("d/noise_metric", noise_metric.mean())
        self.log("d/loss", d_loss)
        self.log("d/est_score", est_score.mean())
        self.log("d/noisy_score", noisy_score.mean())
        self.log("d/est_metric", est_metric.mean())
        self.log("d/noisy_metric", noisy_metric.mean())
        self.toggle_optimizer(d_opt)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        self.clip_grad(d_opt)
        d_opt.step()
        self.untoggle_optimizer(d_opt)
    
        # Optimize SE model
        if self.global_step % self.n_critic == 0:
            if self.teacher is not None:
                SE_loss, est_wav, spec_mask, mag_mask = self.forward(noisy_wav, noisy_wav, ret_mask=True)
            else:
                SE_loss, est_wav = self.forward(noisy_wav, noisy_wav)
            est_score = self.d_model(torch.abs(self.stft.apply_stft(est_wav)))
            g_loss = F.mse_loss(est_score, torch_one)
            if self.SE_loss_w is not None:
                SE_loss = self.SE_loss_w * SE_loss
                self.log("g/SE_loss", SE_loss)
                self.log("g/g_loss", g_loss)
                g_loss = g_loss + SE_loss
            if self.noise_score is not None:
                _noise_score = self.noise_score if isinstance(self.noise_score, float) else \
                    random.uniform(*list(self.noise_score))
                est_noise = noisy_wav - est_wav
                noise_score = self.d_model(torch.abs(self.stft.apply_stft(est_noise)))
                noise_loss = self.noise_weight * F.mse_loss(noise_score, _noise_score*torch_one)
                self.log("g/noise_loss", noise_loss)
                g_loss = g_loss + noise_loss
            if self.score_increment is not None:
                noisy_score = self.d_model(torch.abs(self.stft.apply_stft(noisy_wav)))
                increment_loss = torch.clamp(est_score - noisy_score + self.score_increment, min=0).mean()
                self.log("g/increment_loss", increment_loss)
                g_loss = g_loss + increment_loss
            if self.teacher is not None:
                teacher_wav = batch['teacher_wav']
                if self.train_padding:
                    n_padding = noisy_wav.shape[-1] - teacher_wav.shape[-1]
                    teacher_wav = F.pad(teacher_wav, (0, n_padding))
                teacher_loss = self.get_teacher_loss(noisy_wav, est_wav, teacher_wav, spec_mask, mag_mask)
                self.log("g/teacher_loss", teacher_loss)
                g_loss = g_loss + teacher_loss
            self.log("g/loss", g_loss)
            self.toggle_optimizer(g_opt)
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.clip_grad(g_opt)
            g_opt.step()
            self.untoggle_optimizer(g_opt)


class UnSE_TripleGAN(UnSE_MetricGAN):
    def __init__(self, conf):
        super().__init__(conf)
        self.d2_model = deepcopy(self.d_model)
        self.d3_model = deepcopy(self.d_model)
        self.d_model_list = [self.d_model, self.d2_model, self.d3_model]
    
    def configure_optimizers(self):
        opt_conf = OmegaConf.to_container(self.conf['optimizer'])
        return (self.get_optimizer(opt_conf['generator'], self.model.parameters()),
                self.get_optimizer(opt_conf['discriminator'], self.d_model.parameters()),
                self.get_optimizer(opt_conf['discriminator'], self.d2_model.parameters()),
                self.get_optimizer(opt_conf['discriminator'], self.d3_model.parameters()))
        
    def training_step(self, batch, batch_idx):
        clean_wav, noisy_wav = batch['clean_wav'], batch['noisy_wav']
        noisy_wav, _ = self.data_augment(noisy_wav, clean_wav)
        if self.train_padding:
            noisy_wav = self.padding(noisy_wav)
        torch_one = torch.ones(noisy_wav.shape[0], 1, device=noisy_wav.device)

        g_opt, d1_opt, d2_opt, d3_opt = self.optimizers()

        # Train discriminators
        for idx, (d_opt, d_model) in enumerate(zip([d1_opt, d2_opt, d3_opt], self.d_model_list)):
            est_wav = self.forward(noisy_wav, train=False).detach()
            est_metric = self.metric_score(est_wav, idx)
            noisy_metric = self.metric_score(noisy_wav, idx)
            est_score = d_model(torch.abs(self.stft.apply_stft(est_wav)))
            noisy_score = d_model(torch.abs(self.stft.apply_stft(noisy_wav)))
            d_loss = F.mse_loss(est_metric, est_score) + F.mse_loss(noisy_metric, noisy_score)
            self.log(f"d{idx+1}/loss", d_loss)
            self.log(f"d{idx+1}/est_score", est_score.mean())
            self.log(f"d{idx+1}/noisy_score", noisy_score.mean())
            self.log(f"d{idx+1}/est_metric", est_metric.mean())
            self.log(f"d{idx+1}/noisy_metric", noisy_metric.mean())
            self.toggle_optimizer(d_opt)
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            self.clip_grad(d_opt)
            d_opt.step()
            self.untoggle_optimizer(d_opt)
            
        # Optimize SE model
        if self.global_step % self.n_critic == 0:
            if self.teacher is not None:
                SE_loss, est_wav, spec_mask, mag_mask = self.forward(noisy_wav, noisy_wav, ret_mask=True)
            else:
                SE_loss, est_wav = self.forward(noisy_wav, noisy_wav)
            est_mag = torch.abs(self.stft.apply_stft(est_wav))
            g_loss = sum(F.mse_loss(d_model(est_mag), torch_one) for d_model in self.d_model_list)
            if self.SE_loss_w is not None:
                SE_loss = self.SE_loss_w * SE_loss
                self.log("g/SE_loss", SE_loss)
                self.log("g/g_loss", g_loss)
                g_loss = g_loss + SE_loss
            if self.teacher is not None:
                teacher_wav = batch['teacher_wav']
                if self.train_padding:
                    n_padding = noisy_wav.shape[-1] - teacher_wav.shape[-1]
                    teacher_wav = F.pad(teacher_wav, (0, n_padding))
                teacher_loss = self.get_teacher_loss(noisy_wav, est_wav, teacher_wav, spec_mask, mag_mask)
                self.log("g/teacher_loss", teacher_loss)
                g_loss = g_loss + teacher_loss
            self.log("g/loss", g_loss)
            self.toggle_optimizer(g_opt)
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            self.clip_grad(g_opt)
            g_opt.step()
            self.untoggle_optimizer(g_opt)


class MetricGAN(GAN_Base, UniSE):    
    def __init__(self, conf):
        conf['mag_data'] = conf.get('mag_data', False)
        conf['bidirectional'] = conf.get('bidirectional', True)
        conf['mag_decoder'] = conf.get('mag_decoder', False)
        conf['spec_decoder'] = conf.get('spec_decoder', True)
        conf['norm'] = conf.get('norm', 'batch')
        conf['teacher'] = None # !!!
        super().__init__(conf)
        self.PESQ_worker = ComputePESQ(conf)

    def metric_score(self, clean_wav, noisy_wav):
        scores = self.PESQ_worker.batch_scores(clean_wav.cpu(), noisy_wav.cpu())
        if -1 in scores:
            return None
        scores = (scores - 1) / 3.5
        score_tensor = torch.tensor(scores).type_as(clean_wav).unsqueeze(-1)
        return score_tensor

    def training_step(self, batch, batch_idx, optimizer_idx):
        clean_wav, noisy_wav, _ = batch
        noisy_wav, clean_wav = self.data_augment(noisy_wav, clean_wav)
        torch_one = torch.ones(clean_wav.shape[0], 1).type_as(noisy_wav).float()
        clean_spec = self.stft.apply_stft(clean_wav)
        clean_mag = torch.abs(clean_spec)

        # train discriminator
        if optimizer_idx == 0:
            est_wav = self.forward(noisy_wav, train=False).detach()
            est_spec = self.stft.apply_stft(est_wav)
            est_mag = torch.abs(est_spec)
            est_score = self.d_model(torch.cat([clean_mag, est_mag], dim=1))
            est_metric = self.metric_score(clean_wav, est_wav)
            if est_metric is None:
                return None

            noisy_spec = self.stft.apply_stft(noisy_wav)
            noisy_mag = torch.abs(noisy_spec)
            noisy_score = self.d_model(torch.cat([clean_mag, noisy_mag], dim=1))
            noisy_metric = self.metric_score(clean_wav, noisy_wav)
            if noisy_metric is None:
                return None

            clean_score = self.d_model(torch.cat([clean_mag, clean_mag], dim=1))
            d_loss = (
                F.mse_loss(clean_score, torch_one)
                + F.mse_loss(est_score, est_metric)
                + F.mse_loss(noisy_score, noisy_metric)
            )
            self.log("d/loss", d_loss)
            self.log("d/clean_score", clean_score.mean())
            self.log("d/est_score", est_score.mean())
            self.log("d/noisy_score", noisy_score.mean())
            self.log("d/est_metric", est_metric.mean())
            self.log("d/noisy_metric", noisy_metric.mean())
            return d_loss

        # train generator
        if optimizer_idx == 1:
            SE_loss, est_wav = self.forward(noisy_wav, clean_wav)
            est_spec = self.stft.apply_stft(est_wav)
            est_mag = torch.abs(est_spec)
            est_score = self.d_model(torch.cat([clean_mag, est_mag], dim=1))
            g_loss = F.mse_loss(est_score, torch_one)
            if self.SE_loss_w:
                SE_loss = self.SE_loss_w * SE_loss
                self.log("g/SE_loss", SE_loss)
                self.log("g/g_loss", g_loss)
                g_loss = g_loss + SE_loss
            self.log("g/loss", g_loss)
            return g_loss


# DOTN: Discriminator-Constrained Optimal Transport Network
import modules.blocks.DOTN_modules as DOTN_modules
class DOTN(BaseSE):
    def __init__(self, conf):
        super().__init__(conf)
        assert conf['mag_compress'] == 'log1p'
        assert conf['loss_fun'] == 'MSELoss'
        self.G = DOTN_modules.Generator()
        self.D = DOTN_modules.Discriminator()
        self.mag_compress = conf['mag_compress']

    def configure_optimizers(self):
        period_D = self.conf.get("period_D", 5)
        period_G = self.conf.get("period_G", 11)
        period_S = self.conf.get("period_S", 2)
        optClass = getattr(torch.optim, self.conf.get('optimizer', 'Adam'))
        opt_D = optClass(self.D.parameters(), lr=self.conf.get('lr_D', 1e-3))
        opt_G = optClass(self.G.parameters(), lr=self.conf.get('lr_G', 1e-5))
        opt_S = optClass(self.G.parameters(), lr=self.conf.get('lr_S', 1e-4))
        opt_G_OT = optClass(self.G.parameters(), lr=self.conf.get('lr_G_OT', 1e-5))
        lr_scheduler = self.conf.get(
            'lr_scheduler', {"factor": 0.1, "patience": 4})
        OT_scheduler = LR_scheduler.ReduceLROnPlateau(opt_G_OT, **lr_scheduler)
        return ({"optimizer": opt_D,  'frequency': period_D},
                {"optimizer": opt_G,  'frequency': period_G},
                {"optimizer": opt_S, 'frequency': period_S},
                {"optimizer": opt_G_OT, "lr_scheduler": {
                    "scheduler": OT_scheduler, "monitor": "valid_loss"},
                 'frequency': 1})

    def _calc_stft(self, x):
        x_stft = self.stft.apply_stft(x).squeeze(1).transpose(1, 2)
        x_mag = torch.abs(x_stft)
        if self.mag_compress == 'log1p':
            x_mag = torch.log1p(x_mag)
        x_phase = torch.angle(x_stft)
        return x_mag, x_phase

    def _calc_istft(self, x_mag, x_phase):
        if self.mag_compress == 'log1p':
            x_mag = torch.expm1(x_mag)
        x_stft = x_mag * torch.exp(1.0j*x_phase)
        x = self.stft.apply_istft(x_stft.transpose(1, 2).unsqueeze(1))
        return x

    def forward(self, noisy_wav, clean_wav=None, train=True):
        # use forward for inference/predictions
        noisy_mag, noisy_phase = self._calc_stft(noisy_wav)
        est_mag = self.G(noisy_mag)
        est_wav = self._calc_istft(est_mag, noisy_phase)
        # est_wav = est_wav[:, 0:noisy_wav.shape[1]] # the wav might be padded
        if train is True:
            clean_mag, _ = self._calc_stft(clean_wav)
            loss = self.loss_fun(est_mag, clean_mag)
            return est_wav, loss
        else:
            return est_wav

    def training_step(self, batch, batch_idx, optimizer_idx):
        # s: source domain, t: target domain
        s_clean_wav, s_noisy_wav, t_noisy_wav, _, _ = batch

        # stft, get magnitude
        s_noisy_mag, _ = self._calc_stft(s_clean_wav)
        s_clean_mag, _ = self._calc_stft(s_noisy_wav)
        t_noisy_mag, _ = self._calc_stft(t_noisy_wav)

        # eatimate the magintude
        s_est_mag = self.G(s_noisy_mag)
        t_est_mag = self.G(t_noisy_mag)

        # opt_D
        if optimizer_idx == 0:
            d_loss = -torch.mean(self.D(s_clean_mag)) + torch.mean(self.D(t_est_mag.detach()))
            self.log("train/d_loss", d_loss)  # WGAN, D loss
            return d_loss
        # opt_G
        if optimizer_idx == 1:
            g_loss = -torch.mean(self.D(t_est_mag))
            self.log("train/g_loss", g_loss)  # WGAN, G loss
            return g_loss
        # opt_S
        if optimizer_idx == 2:
            s_loss = self.loss_fun(s_est_mag, s_clean_mag)
            self.log("train/s_loss", s_loss)  # source domain loss
            return s_loss
        # opt_G_OT
        if optimizer_idx == 3:
            ot_loss = self.OT_loss(s_noisy_mag, t_noisy_mag, s_clean_mag, t_est_mag)
            self.log("train/ot_loss", ot_loss)  # OT loss
            return ot_loss

    def OT_loss(self, s_noisy_mag, t_noisy_mag, s_clean_mag, t_est_mag):
        import ot
        N = s_noisy_mag.shape[0]
        C0 = torch.cdist(s_noisy_mag.reshape((N, -1)), t_noisy_mag.reshape((N, -1)), p=2)
        C1 = torch.cdist(s_clean_mag.reshape((N, -1)), t_est_mag.reshape((N, -1)), p=2)
        alpha = 1  # OT source weight in loss
        beta = 1   # OT target weight in loss
        C = alpha * C0 + beta * C1
        γ = ot.emd(torch.tensor(ot.unif(N)).type_as(C), torch.tensor(ot.unif(N)).type_as(C), C)
        loss = torch.sum(γ * C)
        return loss
