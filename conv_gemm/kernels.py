import triton
import triton.language as tl


@triton.jit
def _img2col_kernel_2d(
    x_ptr, cols_ptr,
    N, C, H, W,
    KH, KW, SH, SW, PH, PW, DH, DW,
    H_OUT, W_OUT, K_TOTAL,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_cn, stride_ck, stride_cl,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    L = H_OUT * W_OUT
    # индекс элемента батча
    n = pid // L
    # индекс окна (пикселя)
    l = pid % L
    # координаты окна h w выходного тензора
    oh = l // W_OUT
    ow = l % W_OUT
    # координаты окна во входном тензора
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW
    # смещение по батчу во входном тензоре
    base_xn = n * stride_xn
     # cмещение в выходном буфере
    base_cols_nl = n * stride_cn + l * stride_cl

    # цикл по кернелу с заапасом (попоробовать адаптировать блок х)
    k0 = 0
    while k0 < K_TOTAL:
        k = k0 + tl.arange(0, BLOCK_X)
        k_mask = k < K_TOTAL
        # позиция в плоском 2д кернеле
        r = k % (KH * KW)
        # номер входного канала
        c = k // (KH * KW)
        # вертикальн и гориз координата в ядре
        kh = r // KW
        kw = r % KW
        # координаты пикселя окна во входе
        ih = ih0 + kh * DH
        iw = iw0 + kw * DW

        in_mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & k_mask
        # смещение при чтении входного тензора
        x_off = base_xn + c * stride_xc + ih * stride_xh + iw * stride_xw
        vals = tl.load(x_ptr + x_off, mask=in_mask, other=0)
        # смещение при записи выходного тензора
        cols_off = base_cols_nl + k * stride_ck
        tl.store(cols_ptr + cols_off, vals, mask=k_mask)

        k0 += BLOCK_X

@triton.jit
def _col2img2_kernel_2d(
        dx_ptr,
        dcols_ptr,
        N, C, H, W,
        KH, KW, SH, SW, PH, PW, DH, DW,
        H_OUT, W_OUT, K_TOTAL,
        sxn, sxc, sxh, sxw,
        scn, sck, scl,
        BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    L = H_OUT * W_OUT
    n = pid // L
    l = pid % L
    oh = l // W_OUT
    ow = l % W_OUT
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    base_dx_n = n * sxn
    base_dcols_nl = n * scn + l * scl
    k0 = 0
    while k0 < K_TOTAL:
        k = k0 + tl.arange(0, BLOCK_X)
        k_mask = k < K_TOTAL

        r = k % (KH * KW)
        c = k // (KH * KW)
        kh = r // KW
        kw = r % KW

        ih = ih0 + kh * DH
        iw = iw0 + kw * DW

        inb = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & k_mask
        # грузим  из теензора развернутому [n, k, l]
        dval = tl.load(dcols_ptr + (base_dcols_nl + k * sck), mask=k_mask, other=0)
        tl.atomic_add(dx_ptr + (base_dx_n + c * sxc + ih * sxh + iw * sxw), dval, mask=inb)

        k0 += BLOCK_X

@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    lda, ldb, ldc,                       # row-major: lda=K, ldb=N, ldc=N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_FP16: tl.constexpr,              # хранить C в fp16 или fp32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BN]
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)            # [BK]
        mask_k = offs_k < K

        a = tl.load(a_ptr + (offs_m[:, None] * lda + offs_k[None, :]),
                    mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptr + (offs_k[:, None] * ldb + offs_n[None, :]),
                    mask=mask_k[:, None] & mask_n[None, :], other=0)

        acc += tl.dot(a, b, out_dtype=tl.float32)
        k0 += BLOCK_K

    c = acc.to(tl.float16) if OUT_FP16 else acc
    tl.store(c_ptr + (offs_m[:, None] * ldc + offs_n[None, :]),
             c, mask=mask_m[:, None] & mask_n[None, :])