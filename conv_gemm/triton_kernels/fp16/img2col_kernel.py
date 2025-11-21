import triton
import triton.language as tl


@triton.jit
def img2col_kernel(
    x_ptr, cols_ptr,
    N, Cin, H, W,
    Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
    Ho, Wo,
    sN, sC, sH, sW,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAST_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    M = N * Ho * Wo
    mask_m = offs_m < M
    mask_k = offs_k < K

    n  = offs_m // (Ho * Wo)
    t  = offs_m %  (Ho * Wo)
    ho = t // Wo
    wo = t %  Wo
    n  = n[:, None]; ho = ho[:, None]; wo = wo[:, None]

    cin = offs_k // (Kh * Kw)
    r   = offs_k %  (Kh * Kw)
    kh  = r // Kw
    kw  = r %  Kw
    cin = cin[None, :]; kh = kh[None, :]; kw = kw[None, :]

    ih = ho * Sh - Ph + kh * Dh
    iw = wo * Sw - Pw + kw * Dw

    inb = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_m[:, None] & mask_k[None, :]

    ptr_x = x_ptr + n*sN + cin*sC + ih*sH + iw*sW
    vals = tl.load(ptr_x, mask=inb, other=0)
    if CAST_FP16:
        vals = vals.to(tl.float16)

    ptr_cols = cols_ptr + (offs_m[:, None]*K + offs_k[None, :])
    tl.store(ptr_cols, vals, mask=(mask_m[:, None] & mask_k[None, :]))
