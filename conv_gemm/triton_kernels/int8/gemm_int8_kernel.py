import triton
import triton.language as tl
import torch


@triton.jit
def gemm_int8_tc_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C[M, N] = A[M, K] @ B[K, N]
    A, B: int8, аккумулируем в int32.
    Требования: K % 4 == 0, BLOCK_K % 4 == 0.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_a,
            other=0,
        )
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_b,
            other=0,
        )

        # гарантируем int8 (на случай странных типов)
        a = a.to(tl.int8)
        b = b.to(tl.int8)

        # ВАЖНО: НИКАКОГО input_precision тут
        acc += tl.dot(
            a, b,
            out_dtype=tl.int32,
        )

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


def gemm_int8_tc(
    A_q: torch.Tensor,   # [M,K] int8
    B_q: torch.Tensor,   # [K,N] int8
    *,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """
    Совместимая с твоей gemm_int8() версия.
    Возвращает C_i32: [M,N] int32.
    """
    assert A_q.is_cuda and B_q.is_cuda, "A_q и B_q должны лежать на CUDA"
    assert A_q.dtype == torch.int8 and B_q.dtype == torch.int8, "Оба тензора должны быть int8"

    if not A_q.is_contiguous():
        A_q = A_q.contiguous()
    if not B_q.is_contiguous():
        B_q = B_q.contiguous()

    M, K1 = A_q.shape
    K2, N = B_q.shape
    assert K1 == K2, f"K mismatch: {K1} vs {K2}"

    assert K1 % 4 == 0, f"K={K1} must be divisible by 4 for INT8 dot"
    assert BLOCK_K % 4 == 0, f"BLOCK_K={BLOCK_K} must be divisible by 4"

    C_i32 = torch.empty((M, N), dtype=torch.int32, device=A_q.device)

    a_m, a_k = A_q.stride()
    b_k, b_n = B_q.stride()
    c_m, c_n = C_i32.stride()

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    gemm_int8_tc_kernel[grid](
        A_q, B_q, C_i32,
        M, N, K1,
        a_m, a_k,
        b_k, b_n,
        c_m, c_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return C_i32
