import triton
import triton.language as tl
import torch

INT8_I2C_BLOCK_M = 64
INT8_I2C_BLOCK_K = 32


@triton.jit
def img2col_int8_kernel(
    x_ptr, cols_ptr,
    N, Cin, H, W,
    Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
    Ho, Wo,
    sN, sC, sH, sW,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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

    ptr_x = x_ptr + n * sN + cin * sC + ih * sH + iw * sW

    # ВАЖНО: other того же dtype, что и хотим грузить
    zeros = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int8)
    vals = tl.load(ptr_x, mask=inb, other=zeros)

    ptr_cols = cols_ptr + (offs_m[:, None] * K + offs_k[None, :])
    tl.store(ptr_cols, vals, mask=(mask_m[:, None] & mask_k[None, :]))



def img2col_int8(
    x_q: torch.Tensor,
    Kh: int, Kw: int,
    Sh: int, Sw: int,
    Ph: int, Pw: int,
    Dh: int, Dw: int,
    BLOCK_M: int = INT8_I2C_BLOCK_M,
    BLOCK_K: int = INT8_I2C_BLOCK_K,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """
    x_q: [N, Cin, H, W], int8
    return cols_q: [M, K], int8, (Ho, Wo)
    """
    assert x_q.is_cuda and x_q.dtype == torch.int8
    N, Cin, H, W = x_q.shape

    Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
    Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
    M = N * Ho * Wo
    K = Cin * Kh * Kw

    cols_q = torch.empty((M, K), device=x_q.device, dtype=torch.int8)
    sN, sC, sH, sW = x_q.stride()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    img2col_int8_kernel[grid](
        x_q, cols_q,
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
    return cols_q, (Ho, Wo)
