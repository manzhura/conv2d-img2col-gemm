# === TritonConv2d — nn.Module над твоим TritonConv2dFn ===
import math
import torch, triton
from conv_gemm.operators.triton_conv2d_fp32_fn import TritonConv2dFn



def _force_strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    except Exception:
        pass

_force_strict_fp32()


class TritonConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, bias=True,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        NUM_WARPS=4, NUM_STAGES=2,
        precision_mode="fp32",
        use_weight_shadow=True
    ):
        super().__init__()
        if isinstance(kernel_size, int): kh = kw = kernel_size
        else: kh, kw = kernel_size
        if isinstance(stride, int):   stride = (stride, stride)
        if isinstance(padding, int):  padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kh, kw)
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = torch.nn.Parameter(torch.empty(out_channels)) if bias else None

        # Инициализация как в Conv2d
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * kh * kw
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Тюнинг кернелов
        self.BLOCK_M   = BLOCK_M
        self.BLOCK_N   = BLOCK_N
        self.BLOCK_K   = BLOCK_K
        self.NUM_WARPS = NUM_WARPS
        self.NUM_STAGES = NUM_STAGES

        self.precision_mode = None
        self.use_weight_shadow = use_weight_shadow
        self.register_buffer("weight_fp16_shadow", None, persistent=False)
        self.set_precision(precision_mode)

    # публичный переключатель режимов
    def set_precision(self, mode):
        if mode not in ("fp32", "fp16_runtime", "fp16_infer"):
            raise ValueError("precision_mode must be 'fp32' | 'fp16_runtime' | 'fp16_infer'")
        self.precision_mode = mode

        with torch.no_grad():
            if mode == "fp32":
                self.weight.data = self.weight.data.float()
                if self.bias is not None:
                    self.bias.data = self.bias.data.float()
                self.weight_fp16_shadow = None

            elif mode == "fp16_infer":
                self.weight.data = self.weight.data.half()
                if self.bias is not None:
                    self.bias.data = self.bias.data.float()
                self.weight_fp16_shadow = None

            elif mode == "fp16_runtime":
                self.weight.data = self.weight.data.float()
                if self.bias is not None:
                    self.bias.data = self.bias.data.float()
                if self.use_weight_shadow:
                    self.weight_fp16_shadow = self.weight.detach().half()

    def _refresh_shadow_if_needed(self):
        if self.precision_mode == "fp16_runtime" and self.use_weight_shadow:
            self.weight_fp16_shadow = self.weight.detach().half()

    def forward(self, x):
        assert x.is_cuda and self.weight.is_cuda

        if self.precision_mode == "fp32":
            x_in = x.float()
            w_in = self.weight.float()
            I2C_FP16 = False; GEMM_FP16 = False

        elif self.precision_mode == "fp16_infer":
            x_in = x.half()
            w_in = self.weight.half()
            I2C_FP16 = GEMM_FP16 = True

        elif self.precision_mode == "fp16_runtime":
            x_in = x.half()
            self._refresh_shadow_if_needed()
            w_in = self.weight_fp16_shadow
            I2C_FP16 = GEMM_FP16 = True

        y = TritonConv2dFn.apply(
            x_in, w_in, self.bias,
            self.stride, self.padding, self.dilation,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            self.NUM_WARPS, self.NUM_STAGES,
            I2C_FP16, GEMM_FP16
        )

        if self.precision_mode == "fp32":
            return y
        elif self.precision_mode == "fp16_infer":
            return y
        elif self.precision_mode == "fp16_runtime":
            return y.float()