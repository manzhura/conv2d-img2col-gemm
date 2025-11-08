import math
import time
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_hw_out(h: int, w: int,
                   kH: int, kW: int,
                   sH: int, sW: int,
                   pH: int, pW: int,
                   dH: int, dW: int) -> Tuple[int, int]:
    H_out = (h + 2*pH - dH*(kH-1) - 1)//sH + 1
    W_out = (w + 2*pW - dW*(kW-1) - 1)//sW + 1
    return H_out, W_out

def im2col_unfold(x: torch.Tensor,
                  kernel_size: Tuple[int, int],
                  stride: Tuple[int, int],
                  padding: Tuple[int, int],
                  dilation: Tuple[int, int]) -> torch.Tensor:
    """
    x: [N, C, H, W]
    return col: [N, C*kH*kW, H_out*W_out]
    """
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    col = F.unfold(x, kernel_size=(kH, kW), dilation=(dH, dW),
                   padding=(pH, pW), stride=(sH, sW))
    return col  # [N, K, L]

def gemm_weight_col(weight: torch.Tensor,
                    col: torch.Tensor,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    weight: [C_out, C_in, kH, kW]
    col:    [N, C_in*kH*kW, L]
    return: [N, C_out, L]
    """
    N, K, L = col.shape
    C_out, C_in, kH, kW = weight.shape
    assert K == C_in * kH * kW, "K несовместим с весами"
    Wm = weight.view(C_out, K)                           # [C_out, K]
    y = torch.bmm(Wm.unsqueeze(0).expand(N, -1, -1),     # [N, C_out, K]
                  col)                                   # [N, K, L] -> [N, C_out, L]
    if bias is not None:
        y = y + bias.view(1, -1, 1)
    return y

def im2col_unfold(x: torch.Tensor,
                  kernel_size: Tuple[int, int],
                  stride: Tuple[int, int],
                  padding: Tuple[int, int],
                  dilation: Tuple[int, int]) -> torch.Tensor:
    """
    x: [N, C, H, W]
    return col: [N, C*kH*kW, H_out*W_out]
    """
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    col = F.unfold(x, kernel_size=(kH, kW), dilation=(dH, dW),
                   padding=(pH, pW), stride=(sH, sW))
    return col  # [N, K, L]

def col2im_fold(y: torch.Tensor,
                out_channels: int,
                H_out: int,
                W_out: int) -> torch.Tensor:
    """
    y: [N, C_out, L]
    return: [N, C_out, H_out, W_out]
    """
    N, C_out, L = y.shape
    assert C_out == out_channels
    assert L == H_out * W_out
    return y.view(N, C_out, H_out, W_out)

def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)
class Gem2ColConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias: bool = True):
        super().__init__()
        # --- НОРМАЛИЗАЦИЯ ТОЛЬКО ОДИН РАЗ ---
        kH, kW = _to_2tuple(kernel_size)
        sH, sW = _to_2tuple(stride)
        pH, pW = _to_2tuple(padding)
        dH, dW = _to_2tuple(dilation)

        # (опционально, но полезно) привести к int
        kH, kW = int(kH), int(kW)
        sH, sW = int(sH), int(sW)
        pH, pW = int(pH), int(pW)
        dH, dW = int(dH), int(dW)

        self.in_c  = in_channels
        self.out_c = out_channels
        self.kH, self.kW = kH, kW
        self.sH, self.sW = sH, sW
        self.pH, self.pW = pH, pW
        self.dH, self.dW = dH, dW

        # ВАЖНО: здесь все четыре — именно int
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kH, self.kW))
        self.bias   = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_c * self.kH * self.kW
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, +bound)

    @torch.no_grad()
    def _hw_out(self, h: int, w: int) -> Tuple[int, int]:
        return compute_hw_out(h, w, self.kH, self.kW, self.sH, self.sW, self.pH, self.pW, self.dH, self.dW)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) im2col
        col = im2col_unfold(
            x,
            kernel_size=(self.kH, self.kW),
            stride=(self.sH, self.sW),
            padding=(self.pH, self.pW),
            dilation=(self.dH, self.dW),
        )  # [N, K, L]

        # 2) GEMM
        y = gemm_weight_col(self.weight, col, self.bias)  # [N, C_out, L]

        # 3) col2im (просто reshape)
        N, _, H, W = x.shape
        H_out, W_out = self._hw_out(H, W)
        out = col2im_fold(y, self.out_c, H_out, W_out)
        return out