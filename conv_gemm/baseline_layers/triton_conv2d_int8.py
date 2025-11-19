import math
import torch
import torch.nn as nn

from conv_gemm.baseline_operators.triton_conv2d_int8_fn import TritonConv2dInt8Fn


class TritonConv2dINT8(nn.Module):
    """
    Обёртка над TritonConv2dInt8Fn:
    - хранит INT8-веса и их scale
    - хранит scale для активаций
    - НЕ занимается квантизацией внутри forward
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
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

        # INT8 веса (квантизованные извне)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kh, kw, dtype=torch.int8),
            requires_grad=False,
        )

        # Один скалярный scale для весов
        self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
        # zp для симметрии не используется, но оставим для совместимости
        self.register_buffer("weight_zp", torch.tensor(0, dtype=torch.int32))

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.bias = None

        # Scale для активаций (один скаляр, калибруется извне)
        self.register_buffer("act_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("act_zp", torch.tensor(0, dtype=torch.int32))

    # ==== ЗАГРУЗКА КВАНТ-ПАРАМОВ (вызов из PTQ-калибратора) ====
    @torch.no_grad()
    def load_quant_params(self,
                          w_q: torch.Tensor,
                          w_scale: torch.Tensor,
                          act_scale: torch.Tensor,
                          bias: torch.Tensor | None = None):
        """
        Сюда приходят УЖЕ ГОТОВЫЕ:
          - w_q: int8 веса формы [Cout, Cin, Kh, Kw]
          - w_scale: скалярный scale для весов
          - act_scale: скалярный scale для активаций
        НИКАКОЙ квантизации здесь нет — только copy_ внутрь слоя.
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

    # ==== ИНФЕРЕНС ====
    def forward(self, x_q: torch.Tensor):
        """
        СЮДА ПРИХОДИТ УЖЕ КВАНТИЗОВАННЫЙ x_q (int8),
        сконверченный с использованием self.act_scale.
        НИКАКОЙ КВАНТИЗАЦИИ ВНУТРИ НЕ ДЕЛАЕМ.
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

    # ==== УДОБНЫЙ ХЕЛПЕР ДЛЯ КВАНТИЗАЦИИ ВХОДА С УЖЕ ЗАДАННЫМ act_scale ====
    @torch.no_grad()
    def quantize_input(self, x_fp: torch.Tensor) -> torch.Tensor:
        """
        """
        x_fp = x_fp.to(torch.float32)
        s = float(self.act_scale)
        # Здесь НЕТ пересчёта scale – только "apply" уже известного
        x_q = torch.clamp(torch.round(x_fp / s), -128, 127).to(torch.int8)
        return x_q