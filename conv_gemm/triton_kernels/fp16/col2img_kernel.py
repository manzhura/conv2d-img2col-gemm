import triton
import triton.language as tl


@triton.jit
def col2img_kernel(
    cols_ptr, x_ptr,
    N, Cin, H, W,
    Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
    Ho, Wo,
    sN, sC, sH, sW,
    K,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # индекс блока по M и K
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # локальные оффсеты в плитке
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    M = N * Ho * Wo

    # маски границ
    mask_m = offs_m < M
    mask_k = offs_k < K

    # раскладываем m → (n, ho, wo)
    n  = offs_m // (Ho * Wo)
    t  = offs_m %  (Ho * Wo)
    ho = t // Wo
    wo = t %  Wo
    n  = n[:, None]; ho = ho[:, None]; wo = wo[:, None]

    # раскладываем k → (cin, kh, kw)
    cin = offs_k // (Kh * Kw)
    r   = offs_k %  (Kh * Kw)
    kh  = r // Kw
    kw  = r %  Kw
    cin = cin[None, :]; kh = kh[None, :]; kw = kw[None, :]

    # вычисляем координаты в исходном изображении
    ih = ho * Sh - Ph + kh * Dh
    iw = wo * Sw - Pw + kw * Dw
    # проверяем попадание в границы x
    inb = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_m[:, None] & mask_k[None, :]
    # читаем патчи  из cols
    ptr_cols = cols_ptr + (offs_m[:, None]*K + offs_k[None, :])
    vals = tl.load(ptr_cols, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)
    # адресуем x и делаем atomic_add
    ptr_x = x_ptr + n*sN + cin*sC + ih*sH + iw*sW
    tl.atomic_add(ptr_x, vals, mask=inb)
