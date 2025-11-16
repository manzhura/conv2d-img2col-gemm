import triton
import triton.language as tl
import torch

INT32_C2I_BLOCK_M = 64
INT32_C2I_BLOCK_K = 32


@triton.jit
def col2img_int32_kernel(
    cols_ptr, x_ptr,
    N, Cin, H, W,
    Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
    Ho, Wo,
    sN, sC, sH, sW,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    cols_ptr: int32 [M,K]
    x_ptr:   int32 [N, Cin, H, W]
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    M = N * Ho * Wo
    mask_m = offs_m < M
    mask_k = offs_k < K

    # offs_m -> (n, ho, wo)
    n  = offs_m // (Ho * Wo)
    t  = offs_m %  (Ho * Wo)
    ho = t // Wo
    wo = t %  Wo

    n  = n[:, None]
    ho = ho[:, None]
    wo = wo[:, None]

    # offs_k -> (cin, kh, kw)
    cin = offs_k // (Kh * Kw)
    r   = offs_k %  (Kh * Kw)
    kh  = r // Kw
    kw  = r %  Kw

    cin = cin[None, :]
    kh  = kh[None, :]
    kw  = kw[None, :]

    ih = ho * Sh - Ph + kh * Dh
    iw = wo * Sw - Pw + kw * Dw

    inb = (
        (ih >= 0) & (ih < H) &
        (iw >= 0) & (iw < W) &
        mask_m[:, None] & mask_k[None, :]
    )

    ptr_cols = cols_ptr + (offs_m[:, None] * K + offs_k[None, :])
    vals_i32 = tl.load(
        ptr_cols,
        mask=(mask_m[:, None] & mask_k[None, :]),
        other=0,
    ).to(tl.int32)

    ptr_x = x_ptr + n * sN + cin * sC + ih * sH + iw * sW
    tl.atomic_add(ptr_x, vals_i32, mask=inb)


def col2img_int32(
    cols_i32: torch.Tensor,
    N: int, Cin: int, H: int, W: int,
    Kh: int, Kw: int,
    Sh: int, Sw: int,
    Ph: int, Pw: int,
    Dh: int, Dw: int,
    BLOCK_M: int = INT32_C2I_BLOCK_M,
    BLOCK_K: int = INT32_C2I_BLOCK_K,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """
    cols_i32: [M,K] int32 (например, результат GEMM-grad по cols)
    return x_i32: [N, Cin, H, W] int32 (потом сам переведёшь в float и масштаб)
    """
    assert cols_i32.is_cuda and cols_i32.dtype == torch.int32
    device = cols_i32.device

    Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
    Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1

    M, K = cols_i32.shape
    assert M == N * Ho * Wo
    assert K == Cin * Kh * Kw

    x_i32 = torch.zeros((N, Cin, H, W), device=device, dtype=torch.int32)
    sN, sC, sH, sW = x_i32.stride()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    col2img_int32_kernel[grid](
        cols_i32, x_i32,
        N, Cin, H, W,
        Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
        Ho, Wo,
        sN, sC, sH, sW,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_i32
