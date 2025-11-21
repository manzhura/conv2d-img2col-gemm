import torch, triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_a, other=0
        )
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_b, other=0
        )

        if USE_FP16:
            a = a.to(tl.float16)
            b = b.to(tl.float16)

        # acc  fp16
        acc += tl.dot(a, b, allow_tf32=False)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def triton_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        use_fp16: bool = False,
        BLOCK_M: int | None = None,
        BLOCK_N: int | None = None,
        BLOCK_K: int | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
) -> torch.Tensor:

    assert A.is_cuda and B.is_cuda
    assert A.dim() == 2 and B.dim() == 2
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"K mismatch: {K1} vs {K2}"

    if use_fp16:
        A = A.to(torch.float16, copy=False).contiguous()
        B = B.to(torch.float16, copy=False).contiguous()
    else:
        A = A.to(torch.float32, copy=False).contiguous()
        B = B.to(torch.float32, copy=False).contiguous()

    C = torch.empty((M, N), device=A.device, dtype=torch.float32).contiguous()

    a_m, a_k = A.stride()
    b_k, b_n = B.stride()
    c_m, c_n = C.stride()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid](
        A, B, C,
        M, N, K1,
        a_m, a_k,
        b_k, b_n,
        c_m, c_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        USE_FP16=use_fp16,
        num_warps=num_warps, num_stages=num_stages
    )
    return C
