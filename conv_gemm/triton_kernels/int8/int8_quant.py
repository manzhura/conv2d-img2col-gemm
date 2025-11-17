import torch
from typing import Tuple


@torch.no_grad()
def quantize_int8_sym(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Симметричная ПЕР-ТЕНЗОРНАЯ квантизация в int8.

        x_q = round(x / s),   x_q ∈ [-128, 127]
        s   = max(|x|) / 127

    Аргументы:
        x   : входной тензор (любого float dtype, на любом девайсе)
        eps : защита от деления на ноль

    Возвращает:
        x_q   : тензор того же shape, dtype=torch.int8
        scale : скаляр-тензор (torch.float32) на том же девайсе, что и x
    """
    if not x.is_floating_point():
        x = x.float()
    else:
        x = x.to(torch.float32, copy=False)

    max_abs = x.abs().max()
    # edge case: полностью нулевой тензор
    if max_abs < eps:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        x_q = torch.zeros_like(x, dtype=torch.int8)
        return x_q, scale

    scale = max_abs / 127.0
    inv_scale = 1.0 / (scale + eps)

    x_q = torch.clamp((x * inv_scale).round(), -128, 127).to(torch.int8)
    return x_q, scale


@torch.no_grad()
def quantize_int8_sym_per_channel(
    x: torch.Tensor,
    dim: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Симметричная ПЕР-КАНАЛЬНАЯ квантизация в int8 по указанной размерности.

    Типичный кейс:
        - для Conv2d весов: (Cout, Cin, Kh, Kw), dim=0 → per-output-channel
        - для Linear весов: (Cout, Cin), dim=0 → per-row

    Аргументы:
        x   : float-тензор
        dim : размерность, по которой считаем max(|x|) для каждого канала
        eps : защита от деления на ноль

    Возвращает:
        x_q   : int8-тензор того же shape, что x
        scale : float32-тензор shape = x.shape[dim] (на том же девайсе)
    """
    if not x.is_floating_point():
        x = x.float()
    else:
        x = x.to(torch.float32, copy=False)

    # max_abs по каналу
    max_abs = x.abs().amax(dim=dim, keepdim=True)  # shape: broadcastable

    # защита от нулей
    max_abs_clamped = torch.clamp(max_abs, min=eps)
    scale = max_abs_clamped / 127.0
    inv_scale = 1.0 / (scale + eps)

    x_q = torch.clamp((x * inv_scale).round(), -128, 127).to(torch.int8)

    # приводим scale к вектору по dim (убираем keepdim)
    scale_vec = scale.squeeze(dim=dim).to(torch.float32)
    return x_q, scale_vec


@torch.no_grad()
def dequantize_int8(x_q: torch.Tensor, scale: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    Обратное преобразование int8 → float32:

        x_fp = x_q * s

    Аргументы:
        x_q  : int8-тензор
        scale: либо скаляр (per-tensor), либо 1D-вектор (per-channel)
        dim  : если scale — вектор и per-channel, то dim = размерность канала

    Возвращает:
        x_fp : float32-тензор
    """
    assert x_q.dtype == torch.int8, "x_q должен быть int8"

    x_i32 = x_q.to(torch.int32)

    if scale.ndim == 0:
        # per-tensor
        return x_i32.to(torch.float32) * scale.to(torch.float32)

    # per-channel
    assert dim is not None, "Для per-channel dequantization нужно указать dim"
    # приводим scale к broadcastable виду
    view_shape = [1] * x_i32.ndim
    view_shape[dim] = scale.shape[0]
    scale_bc = scale.view(*view_shape).to(torch.float32)

    return x_i32.to(torch.float32) * scale_bc