import triton
import triton.language as tl
import torch

@triton.jit
def gemm_int8_tc_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K_pad,
    sA_m, sA_k,
    sB_k, sB_n,
    sC_m, sC_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

    for k0 in range(0, K_pad, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptr = A_ptr + offs_m[:, None] * sA_m + offs_k[None, :] * sA_k
        b_ptr = B_ptr + offs_k[:, None] * sB_k + offs_n[None, :] * sB_n

        a = tl.load(a_ptr, mask=(offs_m[:, None]<M)&(offs_k[None,:]<K_pad), other=0)
        b = tl.load(b_ptr, mask=(offs_k[:,None]<K_pad)&(offs_n[None,:]<N), other=0)

        acc += tl.dot(a, b)

    c_ptr = C_ptr + offs_m[:, None] * sC_m + offs_n[None,:] * sC_n
    tl.store(c_ptr, acc,
             mask=(offs_m[:,None] < M) &
                  (offs_n[None, :] < N))


def gemm_int8_tc(A_q, B_q,
                 BLOCK_M, BLOCK_N, BLOCK_K,
                 num_warps, num_stages):
    M, K_pad = A_q.shape
    K_pad2, N = B_q.shape
    assert K_pad == K_pad2

    C = torch.empty((M, N), dtype=torch.int32, device=A_q.device)

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    gemm_int8_tc_kernel[grid](
        A_q, B_q, C,
        M, N, K_pad,
        A_q.stride(0), A_q.stride(1),
        B_q.stride(0), B_q.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return C
