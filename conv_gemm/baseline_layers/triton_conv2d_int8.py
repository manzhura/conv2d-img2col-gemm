import math
import torch
import torch.nn as nn

from conv_gemm.baseline_operators.triton_conv2d_int8_fn import TritonConv2dInt8Fn
from conv_gemm.triton_kernels.int8.int8_quant import quantize_int8_sym



class TritonConv2dINT8(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
    ):
        super().__init__()

        # --- нормализуем аргументы kernel/stride/padding/dilation ---
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
        # === инициализация весов во float, КВАНТИЗАЦИЯ ОДИН РАЗ ===
        w_f = torch.empty(out_channels, in_channels, kh, kw)
        torch.nn.init.kaiming_uniform_(w_f, a=math.sqrt(5))

        w_q, weight_scale  = quantize_int8_sym(w_f)

        # int8-параметр (forward-only, градиенты не нужны)
        self.weight = torch.nn.Parameter(w_q, requires_grad=False)
        # scale для весов — храним как буфер
        self.register_buffer(
            "weight_scale",
            torch.tensor(weight_scale, dtype=torch.float32),
            persistent=True,
        )
        # === bias: FP32 ===
        if bias:
            fan_in = in_channels * kh * kw
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty(out_channels)
            torch.nn.init.uniform_(b, -bound, bound)
            self.bias = torch.nn.Parameter(b.float(), requires_grad=False)
        else:
            self.bias = None

        # # Тюнинг кернелов
        # self.BLOCK_M = BLOCK_M
        # self.BLOCK_N = BLOCK_N
        # self.BLOCK_K = BLOCK_K
        # self.NUM_WARPS = NUM_WARPS
        # self.NUM_STAGES = NUM_STAGES

        # Маски разреженности
        self.register_buffer(
            "channel_mask",
            torch.ones(out_channels, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "input_channel_mask",
            torch.ones(in_channels, dtype=torch.bool),
            persistent=True,
        )

        self.block_size = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda and self.weight.is_cuda

        #    Сигнатура Fn: apply(x_int8, w_int8, stride, padding, dilation)
        y = TritonConv2dInt8Fn.apply(
            x,
            self.weight,
            self.weight_scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )



        # можешь вернуть float32 или .half() — на твой вкус
        return y
