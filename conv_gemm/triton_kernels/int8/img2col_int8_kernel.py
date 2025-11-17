import triton
import triton.language as tl
import torch


@triton.jit
def img2col_int8_kernel(
    x_ptr, cols_ptr,
    N, Cin, H, W,
    Kh, Kw,
    Sh, Sw,
    Ph, Pw,
    Dh, Dw,
    Ho, Wo,
    sN, sC, sH, sW,
    K_real, K_pad,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    M = N * Ho * Wo

    mask_m = offs_m < M
    mask_k = offs_k < K_pad
    mask = mask_m[:, None] & mask_k[None, :]

    # -------------------------
    # decode m → (n, ho, wo)
    # -------------------------
    m = offs_m.to(tl.int32)
    n  = m // (Ho * Wo)
    rem = m - n * (Ho * Wo)
    ho = rem // Wo
    wo = rem - ho * Wo

    n  = n[:, None]
    ho = ho[:, None]
    wo = wo[:, None]

    # -------------------------
    # decode k → (cin, kh, kw)
    # -------------------------
    k = offs_k.to(tl.int32)
    cin = k // (Kh * Kw)
    remk = k - cin * (Kh * Kw)
    kh = remk // Kw
    kw = remk - kh * Kw

    cin = cin[None, :]
    kh  = kh[None, :]
    kw  = kw[None, :]

    # -------------------------
    # compute input coords
    # -------------------------
    ih = ho * Sh - Ph + kh * Dh
    iw = wo * Sw - Pw + kw * Dw

    inside_real = (
        (offs_k[None, :] < K_real) &
        (ih >= 0) & (ih < H) &
        (iw >= 0) & (iw < W)
    ) & mask

    offset = (
        n * sN +
        cin * sC +
        ih * sH +
        iw * sW
    )

    x_ptr_safe = x_ptr + tl.where(inside_real, offset, 0)
    vals = tl.load(x_ptr_safe, mask=inside_real, other=0)

    out_off = offs_m[:, None].to(tl.int64) * K_pad + offs_k[None, :]
    cols_ptr_safe = cols_ptr + tl.where(mask, out_off, 0)

    tl.store(cols_ptr_safe, vals, mask=mask)


def img2col_int8(
    x_q,
    Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
    K_pad,
    BLOCK_M,
    BLOCK_K,
    num_warps,
    num_stages,
):
    N, Cin, H, W = x_q.shape

    Ho = (H + 2*Ph - Dh*(Kh - 1) - 1)//Sh + 1
    Wo = (W + 2*Pw - Dw*(Kw - 1) - 1)//Sw + 1

    M = N * Ho * Wo
    K_real = Cin * Kh * Kw

    cols_q = torch.empty((M, K_pad), dtype=torch.int8, device=x_q.device)

    sN, sC, sH, sW = x_q.stride()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K_pad, BLOCK_K))

    img2col_int8_kernel[grid](
        x_q, cols_q,
        N, Cin, H, W,
        Kh, Kw,
        Sh, Sw,
        Ph, Pw,
        Dh, Dw,
        Ho, Wo,
        sN, sC, sH, sW,
        K_real, K_pad,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages
    )

    return cols_q, (Ho, Wo)
