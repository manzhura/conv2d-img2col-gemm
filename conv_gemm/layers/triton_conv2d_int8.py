import torch
import torch.nn as nn
import math

from conv_gemm.operators.triton_conv2d_int8_fn import TritonConv2dInt8Fn


class TritonConv2dInt8(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True,
                 BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                 NUM_WARPS=4, NUM_STAGES=2):
        super().__init__()

        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kh, kw)
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation

        self.BLOCK_M   = BLOCK_M
        self.BLOCK_N   = BLOCK_N
        self.BLOCK_K   = BLOCK_K
        self.NUM_WARPS = NUM_WARPS
        self.NUM_STAGES = NUM_STAGES

        # веса храним в float (как обычно)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias   = nn.Parameter(torch.empty(out_channels)) if bias else None

        # инициализация как в Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * kh * kw
            bound  = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def init_from_conv(self, conv_ref: nn.Conv2d):
        """
        Удобный хелпер:
        копируем веса/биас из обычного nn.Conv2d (fp32 или fp16).
        """
        self.weight.copy_(conv_ref.weight.float())
        if self.bias is not None and conv_ref.bias is not None:
            self.bias.copy_(conv_ref.bias.float())

    def forward(self, x: torch.Tensor):
        # x может быть fp16 или fp32 — внутри мы всё равно квантим в int8
        assert x.is_cuda and self.weight.is_cuda, "x и weight должны быть на CUDA"

        return TritonConv2dInt8Fn.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            self.NUM_WARPS, self.NUM_STAGES,
        )
