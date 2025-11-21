from dataclasses import dataclass


@dataclass(frozen=True)
class KernelConfig:
    BLOCK_M: int = 0
    BLOCK_N: int = 0   #
    BLOCK_K: int = 0
    NUM_WARPS: int = 0
    NUM_STAGES: int = 0

# img2col INT8
INT8_I2C_CFG = KernelConfig(
    BLOCK_M=32,
    BLOCK_K=64,
    NUM_WARPS=2,
    NUM_STAGES=2,
)

# GEMM INT8
INT8_GEMM_CFG = KernelConfig(
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=64,
    NUM_WARPS=4,
    NUM_STAGES=3,
)

# col2img INT32
INT8_C2I_CFG = KernelConfig(
    BLOCK_M=32,
    BLOCK_K=32,
    NUM_WARPS=4,
    NUM_STAGES=2,
)

# img2col FP16
FP16_I2C_CFG = KernelConfig(
    BLOCK_M=32,
    BLOCK_K=128,
    NUM_WARPS=2,
    NUM_STAGES=2,
)
# gemm FP16
FP16_GEMM_CFG = KernelConfig(
    BLOCK_M=64,
    BLOCK_N=128,
    BLOCK_K=32,
    NUM_WARPS=4,
    NUM_STAGES=3,
)
# col2img FP32
FP32_C2I_CFG = KernelConfig(
    BLOCK_M=32,
    BLOCK_K=32,
    NUM_WARPS=4,
    NUM_STAGES=2,
)
