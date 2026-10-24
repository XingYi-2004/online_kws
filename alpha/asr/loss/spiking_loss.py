import torch
import torch.nn as nn

class SpikingCELoss:
    sequence_loss = False

    def __init__(self, conf):
        self.label_smoothing = conf.get('label_smoothing', 0.0)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def __call__(self, logits, _feats, labels):
        loss = self.loss_fn(logits, labels)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()
        return loss, {'loss': loss.detach(), 'token_acc': acc.detach()}

    def log_stat(self, loss_info):
        return {
            'loss': self._to_float(loss_info.get('loss')),
            'token_acc': self._to_float(loss_info.get('token_acc')),
        }

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().float().mean().item())
        return float(value)

class TETLoss:
    """Temporal Efficient Training (TET) Loss for Spiking Neural Networks."""
    sequence_loss = True
    
    def __init__(self, conf):
        self.label_smoothing = conf.get('label_smoothing', 0.0)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def __call__(self, logits, attention_mask, labels):
        """
        Args:
            logits: sequence logits shaped (B, T, C)
            attention_mask: optional bool mask shaped (B, T), where True marks valid frames
            labels: target labels shaped (B,)
        """
        if logits.dim() != 3:
            raise ValueError(f"TETLoss expects 3D logits (B, T, C), but got {tuple(logits.shape)}. "
                             "Please ensure 'return_sequence_logits=True' in your model config.")

        batch, time, n_classes = logits.shape
        labels = labels.long()
        labels_seq = labels.unsqueeze(1).expand(batch, time)

        if attention_mask is not None:
            if attention_mask.shape != logits.shape[:2]:
                raise ValueError(
                    f"TETLoss expected attention_mask shape {tuple(logits.shape[:2])}, "
                    f"but got {tuple(attention_mask.shape)}"
                )
            valid_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
            logits_flat = logits[valid_mask]
            labels_flat = labels_seq.to(device=logits.device)[valid_mask]
            mask_float = valid_mask.unsqueeze(-1).to(dtype=logits.dtype)
            denom = valid_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype=logits.dtype)
            acc_logits = (logits * mask_float).sum(dim=1) / denom
        else:
            logits_flat = logits.reshape(batch * time, n_classes)
            labels_flat = labels_seq.reshape(-1).to(device=logits.device)
            acc_logits = logits.mean(dim=1)
        
        loss = self.loss_fn(logits_flat, labels_flat)
        
        with torch.no_grad():
            preds = torch.argmax(acc_logits, dim=-1)
            acc = (preds == labels.to(device=preds.device)).float().mean()
            
        return loss, {'loss': loss.detach(), 'token_acc': acc.detach()}

    def log_stat(self, loss_info):
        return {
            'loss': self._to_float(loss_info.get('loss')),
            'token_acc': self._to_float(loss_info.get('token_acc')),
        }

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().float().mean().item())
        return float(value)

class HybridTETLoss:
    """Hybrid Temporal Efficient Training Loss (TET + Global CE) for SNNs."""
    sequence_loss = True
    
    def __init__(self, conf):
        self.label_smoothing = conf.get('label_smoothing', 0.0)
        self.tet_weight = float(conf.get('tet_weight', 0.2))
        if self.tet_weight < 0:
            raise ValueError('tet_weight must be non-negative')

        self.seq_reduce = conf.get('seq_reduce', conf.get('time_reduce', conf.get('reduce', 'sum')))
        if self.seq_reduce not in ('mean', 'sum'):
            raise ValueError(f'HybridTETLoss supports seq_reduce mean/sum, got {self.seq_reduce}')

        self.frame_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        self.seq_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def __call__(self, logits, attention_mask, labels):
        if logits.dim() != 3:
            raise ValueError(f'HybridTETLoss expects 3D logits (B, T, C), got {tuple(logits.shape)}')

        batch, time, n_classes = logits.shape
        labels = labels.to(device=logits.device, dtype=torch.long)
        labels_seq = labels.unsqueeze(1).expand(batch, time)

        if attention_mask is not None:
            if attention_mask.shape != logits.shape[:2]:
                raise ValueError(
                    f'HybridTETLoss expected attention_mask shape {tuple(logits.shape[:2])}, '
                    f'but got {tuple(attention_mask.shape)}'
                )
            valid_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
        else:
            valid_mask = None

        frame_loss = self.frame_loss_fn(
            logits.reshape(batch * time, n_classes),
            labels_seq.reshape(-1),
        ).reshape(batch, time)
        if valid_mask is not None:
            mask_float = valid_mask.to(dtype=frame_loss.dtype)
            denom = mask_float.sum(dim=1).clamp_min(1.0)
            loss_tet = (frame_loss * mask_float).sum(dim=1).div(denom).mean()
        else:
            loss_tet = frame_loss.mean(dim=1).mean()

        acc_logits = self._reduce_sequence(logits, valid_mask)
        loss_seq = self.seq_loss_fn(acc_logits, labels)
        loss = loss_seq + self.tet_weight * loss_tet
        
        with torch.no_grad():
            preds = torch.argmax(acc_logits, dim=-1)
            acc = (preds == labels).float().mean()
            
        return loss, {'loss': loss.detach(), 
                      'loss_tet': loss_tet.detach(), 
                      'loss_seq': loss_seq.detach(), 
                      'token_acc': acc.detach()}

    def _reduce_sequence(self, logits, valid_mask):
        if self.seq_reduce == 'mean':
            if valid_mask is None:
                return logits.mean(dim=1)
            mask = valid_mask.unsqueeze(-1).to(dtype=logits.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (logits * mask).sum(dim=1) / denom

        probs = torch.softmax(logits, dim=-1)
        if valid_mask is not None:
            probs = probs * valid_mask.unsqueeze(-1).to(dtype=probs.dtype)
        return probs.sum(dim=1)

    def log_stat(self, loss_info):
        return {
            'loss': self._to_float(loss_info.get('loss')),
            'loss_tet': self._to_float(loss_info.get('loss_tet')),
            'loss_seq': self._to_float(loss_info.get('loss_seq')),
            'token_acc': self._to_float(loss_info.get('token_acc')),
        }

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().float().mean().item())
        return float(value)
