import torch
import torch.nn.functional as F


@torch.jit.script
def bias_sigmod_ele(y, bias, z):
    return torch.sigmoid(y + bias) * z


@torch.jit.script
def bias_dropout_add(x: torch.Tensor, bias: torch.Tensor, dropmask: torch.Tensor,
                     residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = (x + bias) * F.dropout(dropmask, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_ele_dropout_residual(ab: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
                              dropout_mask: torch.Tensor, Z_raw: torch.Tensor, prob: float,
                              training: bool) -> torch.Tensor:
    return Z_raw + F.dropout(dropout_mask, p=prob, training=training) * (g * (ab + b))
