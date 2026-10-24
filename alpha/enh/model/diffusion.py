import torch
import importlib
from . import model as models
from modules.blocks.blocks import ModelBase
from modules.utils.logging import logger

class DiffAdaptNet(ModelBase):
    ''' 适配不同后端骨干模型的中间层模块
    backbone: NCSNpp, MatchaDecoder
    '''
    def __init__(self, model_conf):
        super().__init__(model_conf)
        self.noisy_cond = model_conf['ode_param'].get('noisy_cond', True)
        self.complex = model_conf.get('complex', True) # complex or mag
        in_channels = (2 if self.noisy_cond else 1) * (2 if self.complex else 1)
        out_channels = 2 if self.complex else 1
        backbone = self.model_conf['backbone']
        # run-time import
        if backbone.startswith('NCSNpp'):
            mod = importlib.import_module('modules.blocks.ncsnpp')
            self.backbone = getattr(mod, backbone)(
                in_channels=in_channels, out_channels=out_channels)
        elif backbone == 'MatchaDecoder':
            mod = importlib.import_module('modules.blocks.matcha_modules')
            self.backbone = getattr(mod, 'Decoder')(
                in_channels=256*in_channels, out_channels=256*out_channels, 
                **self.model_conf['param'])
        elif backbone in ['MIMONetTimeEmbed']:
            self.model_conf.update({'in_channels': in_channels})
            self.backbone = getattr(models, backbone)(self.model_conf)
        else:
            raise NotImplementedError(backbone)
        
    def forward(self, x, t):
        return self.forward_complex(x, t) if self.complex else self.forward_mag(x, t)
            
    def forward_complex(self, x, t):
        # x: shape of [B, 1, F, T] or [B, 2, F, T], complex64
        if self.model_conf['backbone'].startswith('NCSNpp'):
            if self.noisy_cond: # complex64 [B, 2, F, T] -> float32 [B, 4, F, T]
                x = torch.stack((x[:,0,:,:].real, x[:,0,:,:].imag, 
                                x[:,1,:,:].real, x[:,1,:,:].imag), dim=1)
            else: # complex64 [B, 1, F, T] -> float32 [B, 2, F, T]
                x = torch.stack((x[:,0,:,:].real, x[:,0,:,:].imag), dim=1)
            # logger.info(f'x: {x.dtype}, {x.shape}; t: {t.dtype}, {t.shape}')
            vf = self.backbone(x, t) # [B, 2, F, T]
            vf = torch.view_as_complex(vf.permute(0, 2, 3, 1).contiguous())
        elif self.model_conf['backbone'] == 'MatchaDecoder':
            if self.noisy_cond: # complex64 [B, 2, F, T] -> float32 [B, F*4, T]
                x = torch.cat((x[:,0,:,:].real, x[:,0,:,:].imag, 
                            x[:,1,:,:].real, x[:,1,:,:].imag), dim=1)
            else: # complex64 [B, 1, F, T] -> float32 [B, F*2, T]
                x = torch.cat((x[:,0,:,:].real, x[:,0,:,:].imag), dim=1)
            vf = self.backbone(x, t) # [B, F*2, T]
            vf = torch.complex(*torch.chunk(vf, 2, dim=1))
        elif self.model_conf['backbone'] in ['MIMONetTimeEmbed',]:
            if self.noisy_cond: # complex64 [B, 2, F, T] -> float32 [B, 4, F, T]
                x = torch.stack((x[:,0,:,:].real, x[:,0,:,:].imag, 
                                x[:,1,:,:].real, x[:,1,:,:].imag), dim=1)
            else: # complex64 [B, 1, F, T] -> float32 [B, 2, F, T]
                x = torch.stack((x[:,0,:,:].real, x[:,0,:,:].imag), dim=1)
            # logger.info(f'x: {x.dtype}, {x.shape}; t: {t.dtype}, {t.shape}')
            vf = self.backbone(x, t)[0] # [B, 2, F, T]
            vf = torch.view_as_complex(vf.permute(0, 2, 3, 1).contiguous())
        else:
            raise NotImplementedError(self.model_conf['backbone'])
        return vf.unsqueeze(1) # [B, F, T] -> [B, 1, F, T]
    
    def forward_mag(self, x, t):
        # x: shape of [B, 1, F, T] or [B, 2, F, T], float32
        logger.debug(f'x: {x.shape}, t: {t.shape}')
        if self.model_conf['backbone'].startswith('NCSNpp'):
            vf = self.backbone(x, t) # [B, 1, F, T]
        elif self.model_conf['backbone'] == 'MatchaDecoder':
            if self.noisy_cond: # [B, 2, F, T] -> [B, F*2, T]
                x = torch.cat((x[:,0,:,:], x[:,1,:,:]), dim=1)
            else: # [B, 1, F, T] -> [B, F, T]
                x = x.squeeze(1)
            vf = self.backbone(x, t) # [B, F, T]
            vf = vf.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]
        elif self.model_conf['backbone'] in ['MIMONetTimeEmbed',]:
            vf = self.backbone(x, t)[0] # [B, 1, F, T]
        else:
            raise NotImplementedError(self.model_conf['backbone'])
        return vf # [B, 1, F, T]