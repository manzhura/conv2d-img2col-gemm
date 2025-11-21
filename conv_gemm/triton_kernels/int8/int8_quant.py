import torch
from typing import Tuple

QMIN = -128
QMAX = 127

@torch.no_grad()
def quantize_int8_sym_tensor(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Симметричная тензорная квантизация
    """
    x_fp = x.float()

    max_abs = x_fp.abs().max()

    if max_abs < eps:
        scale = torch.tensor(1.0, device=x.device)
        x_q = torch.zeros_like(x_fp, dtype=torch.int8)
    else:
        scale = max_abs / float(QMAX)
        x_q = torch.clamp((x_fp / scale).round(), QMIN, QMAX).to(torch.int8)

    zp = torch.tensor(0.0, device=x.device)
    return x_q, scale, zp