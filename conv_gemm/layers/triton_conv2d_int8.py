import math
import torch
import torch.nn as nn

from conv_gemm.operators.triton_conv2d_int8_fn import TritonConv2dInt8Fn


class TritonConv2dINT8(nn.Module):
    """
    INT8-слой поверх TritonConv2dInt8Fn.

    Режимы:
      - "int8_infer":
          * forward считает через INT8-кернелы (квантизация внутри Function)
          * наружу возвращаем тот же dtype, что у входа x (обычно fp16/fp16)
          * предполагается чистый инференс, градиенты можно игнорировать

      - "int8_runtime":
          * forward такой же (INT8 inside), но результат приводим к float32
          * backward реализован в TritonConv2dInt8Fn через FP32 img2col+GEMM+col2img
          * можно дообучать (QAT-стайл): master-weights в FP32, INT8 только в compute

    Параметр use_weight_shadow пока зарезервирован под будущую реализацию
    внешней квантизации весов (int8-шадоу), но сейчас фактическая квантилка
    живёт внутри TritonConv2dInt8Fn и от этого флага не зависит.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        precision_mode: str = "int8_runtime",
        use_weight_shadow: bool = False,
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

        # --- master weights всегда в FP32 ---
        w_f = torch.empty(out_channels, in_channels, kh, kw)
        torch.nn.init.kaiming_uniform_(w_f, a=math.sqrt(5))
        w_int8 = torch.clamp(w_f, -128, 127).round().to(torch.int8)
        self.register_buffer("weight", w_int8)

        # --- bias ---
        if bias:
            fan_in = in_channels * kh * kw
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty(out_channels)
            torch.nn.init.uniform_(b, -bound, bound)
            # bias в float32, добавляем уже к float-выходу
            self.bias = torch.nn.Parameter(b.float())
        else:
            self.bias = None

        # режим и "тень" (пока без реальной отдельной квантизации)
        if precision_mode not in ("int8_infer", "int8_runtime"):
            raise ValueError("precision_mode must be 'int8_infer' | 'int8_runtime'")
        self.precision_mode = precision_mode
        self.use_weight_shadow = use_weight_shadow

        # # Тюнинг кернелов
        # self.BLOCK_M = BLOCK_M
        # self.BLOCK_N = BLOCK_N
        # self.BLOCK_K = BLOCK_K
        # self.NUM_WARPS = NUM_WARPS
        # self.NUM_STAGES = NUM_STAGES

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
        assert x.is_cuda and self.weight.is_cuda, "x и weight должны быть на CUDA"


        y = TritonConv2dInt8Fn.apply(
            x,               # x (fp16 или fp16) -> внутри квантуется в int8
            self.weight,     # master FP32 weights -> внутри квантуются в int8
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )

        if self.precision_mode == "int8_infer":
            # Инференс: оставляем dtype таким же, как у x (Function уже сделал y.to(x.dtype))
            return y

        elif self.precision_mode == "int8_runtime":
            # QAT/Runtime: наружу всегда float32, даже если вход был fp16
            # (для стабильных градиентов в остальной модели)
            return y.float()
