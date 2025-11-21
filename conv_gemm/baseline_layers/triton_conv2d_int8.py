import math
import torch
import torch.nn as nn

from conv_gemm.baseline_operators.triton_conv2d_int8_fn import TritonConv2dInt8Fn


class TritonConv2dINT8(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, bias=True):
        """
        INT8-свёртка на Triton (инференс-only).
        Возможности:
        • Веса и активации заранее квантованы (PTQ / QAT), внутри слоя квантизации нет.
        • forward: чистый INT8 → INT32 акумуляция → FP32 bias → FP32 выход.
        • Использует TritonConv2dInt8Fn: im2col → INT8 GEMM → col2img.
        • Один скалярный scale для весов и один для активаций.
        • Веса и bias не обучаемые (requires_grad=False).
        """
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

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # weights initialization
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kh, kw, dtype=torch.int8),
            requires_grad=False,
        )

        # weights scale
        self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("weight_zp", torch.tensor(0, dtype=torch.int32))

        #  zero bias initialization
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.bias = None

        # input scale
        self.register_buffer("act_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("act_zp", torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def load_quant_params(self,
                          w_q: torch.Tensor,
                          w_scale: torch.Tensor,
                          act_scale: torch.Tensor,
                          bias: torch.Tensor | None = None):
        """
        Загружает уже готовые квантованные параметры:
        • w_q - int8 веса [Cout, Cin, Kh, Kw]
        • w_scale - scale для весов (скаляр)
        • act_scale - scale для активаций (скаляр)
        • bias - опционально FP32 bias
        """

        assert w_q.dtype == torch.int8, f"w_q must be int8, got {w_q.dtype}"
        assert w_q.shape == self.weight.shape, (
            f"w_q shape {w_q.shape} != layer weight shape {self.weight.shape}"
        )

        self.weight.copy_(w_q)
        self.weight_scale.copy_(w_scale.view(()))
        self.act_scale.copy_(act_scale.view(()))

        if bias is not None:
            assert self.bias is not None, "Layer was created with bias=False"
            assert bias.shape == self.bias.shape, (
                f"bias shape {bias.shape} != layer bias shape {self.bias.shape}"
            )
            self.bias.copy_(bias.float())

    def forward(self, x_q: torch.Tensor):
        """
        INT8-инференс.
        Ожидает:
        • x_q - уже квантованный int8 вход (scale = self.act_scale).
        Выполняет:
        • im2col (int8)
        • INT8×INT8 → INT32 GEMM в Triton
        • добавление FP32 bias
        • col2img → FP32 выход
        Никакого обучения, никакого квантизирования внутри.
        """
        assert x_q.dtype == torch.int8, f"expected int8 input, got {x_q.dtype}"
        assert x_q.is_cuda, "x_q must be on CUDA"

        return TritonConv2dInt8Fn.apply(
            x_q,
            self.weight,
            self.weight_scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.act_scale,
        )

    # helper для PTQ
    @torch.no_grad()
    def quantize_input(self, x_fp: torch.Tensor) -> torch.Tensor:
        """
        Квантование входа по заранее известному act_scale
        x_q = round(x_fp / act_scale).
        scale не пересчитывается.
        """
        x_fp = x_fp.to(torch.float32)
        s = float(self.act_scale)
        x_q = torch.clamp(torch.round(x_fp / s), -128, 127).to(torch.int8)
        return x_q
