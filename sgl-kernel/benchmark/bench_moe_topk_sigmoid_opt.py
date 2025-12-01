"""
Benchmark for the optimized topk_sigmoid_opt kernel.

Compares performance between:
- torch native implementation
- sglang original topk_sigmoid
- sglang optimized topk_sigmoid_opt

The optimization exploits sigmoid's properties:
- Monotonicity: sigmoid(a) > sigmoid(b) iff a > b
- Independence: sigmoid(x_i) only depends on x_i

This allows TopK selection on raw logits, computing sigmoid only for TopK elements.
Expected improvement: Reduces exp() operations from num_experts to topk.
"""
import itertools
import os

import torch
import triton
from sgl_kernel import topk_sigmoid, topk_sigmoid_opt

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def torch_topk_sigmoid_native(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    """Native torch implementation for reference."""
    scores = gating_output.sigmoid()
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        _, topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(1, topk_indices)
    else:
        topk_weights, topk_indices = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


def sglang_topk_sigmoid(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    """Original sglang topk_sigmoid implementation."""
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=renormalize,
        correction_bias=correction_bias,
    )

    return topk_weights, topk_indices


def sglang_topk_sigmoid_opt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    """Optimized sglang topk_sigmoid implementation."""
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=renormalize,
        correction_bias=correction_bias,
    )

    return topk_weights, topk_indices


def get_topk_sigmoid_input(num_tokens, num_experts):
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn((num_experts), dtype=torch.float32, device="cuda")
    return gating_output, correction_bias


def calculate_diff(num_tokens, num_experts, topk, use_bias=False):
    """Verify correctness of optimized implementation."""
    gating_output, correction_bias = get_topk_sigmoid_input(num_tokens, num_experts)
    bias = correction_bias if use_bias else None

    weights_torch, indices_torch = torch_topk_sigmoid_native(
        gating_output.clone(),
        topk,
        True,
        bias.clone() if bias is not None else None,
    )
    weights_opt, indices_opt = sglang_topk_sigmoid_opt(
        gating_output.clone(),
        topk,
        True,
        bias.clone() if bias is not None else None,
    )

    weights_diff = torch.abs(weights_torch - weights_opt).mean().item()
    indices_match = torch.equal(indices_torch, indices_opt)

    bias_str = "with bias" if use_bias else "without bias"
    if (
        torch.allclose(weights_torch, weights_opt, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        print(f"✅ [{bias_str}] Torch and SGLang optimized implementations match")
    else:
        print(
            f"❌ [{bias_str}] Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )


# CI environment uses simplified parameters
if IS_CI:
    num_tokens_range = [128]  # Single value for CI
    num_experts_range = [32]  # Single value for CI
    topk_range = [2]  # Single value for CI
else:
    num_tokens_range = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_experts_range = [32, 64, 128, 256, 12, 512]
    topk_range = [1, 2, 4, 8]

configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


# Benchmark providers: torch, original sglang, optimized sglang
line_vals = ["torch", "sglang", "sglang_opt"]
line_names = ["Torch", "SGLang Original", "SGLang Optimized"]
styles = [("green", "-"), ("blue", "-"), ("red", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="Latency (us)",
        plot_name="topk-sigmoid-opt-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    gating_output, _ = get_topk_sigmoid_input(num_tokens, num_experts)

    if provider == "torch":
        def fn():
            return torch_topk_sigmoid_native(gating_output, topk, True, None)
    elif provider == "sglang":
        def fn():
            return sglang_topk_sigmoid(gating_output, topk, True, None)
    elif provider == "sglang_opt":
        def fn():
            return sglang_topk_sigmoid_opt(gating_output, topk, True, None)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="Latency (us)",
        plot_name="topk-sigmoid-opt-performance-with-bias",
        args={},
    )
)
def benchmark_with_bias(num_tokens, num_experts, topk, provider):
    """Benchmark with correction_bias (optimized kernel falls back to original path)."""
    gating_output, correction_bias = get_topk_sigmoid_input(num_tokens, num_experts)

    if provider == "torch":
        def fn():
            return torch_topk_sigmoid_native(gating_output, topk, True, correction_bias)
    elif provider == "sglang":
        def fn():
            return sglang_topk_sigmoid(gating_output, topk, True, correction_bias)
    elif provider == "sglang_opt":
        def fn():
            return sglang_topk_sigmoid_opt(gating_output, topk, True, correction_bias)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Simplify configs for CI environment
    if IS_CI:
        test_configs = [(20, 32, 2)]  # Single config for CI
    else:
        test_configs = [
            (20, 256, 4),
            (20, 256, 8),
            (20, 12, 4),
            (20, 12, 1),
            (20, 512, 4),
            (20, 512, 1),
        ]

    print("=" * 60)
    print("Correctness verification (without correction_bias)")
    print("=" * 60)
    for num_tokens, num_experts, topk in test_configs:
        calculate_diff(num_tokens, num_experts, topk, use_bias=False)

    print("\n" + "=" * 60)
    print("Correctness verification (with correction_bias)")
    print("=" * 60)
    for num_tokens, num_experts, topk in test_configs:
        calculate_diff(num_tokens, num_experts, topk, use_bias=True)

    print("\n" + "=" * 60)
    print("Performance benchmark (without correction_bias)")
    print("This is where the optimization shines!")
    print("=" * 60)
    benchmark.run(print_data=True)

    print("\n" + "=" * 60)
    print("Performance benchmark (with correction_bias)")
    print("Optimized kernel falls back to original path here.")
    print("=" * 60)
    benchmark_with_bias.run(print_data=True)
