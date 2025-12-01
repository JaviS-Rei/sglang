"""
Test suite for the optimized topk_sigmoid_opt kernel.

The optimization exploits sigmoid's monotonicity and element-wise independence:
- sigmoid(a) > sigmoid(b) iff a > b (monotonicity)
- sigmoid(x_i) only depends on x_i (independence)

This allows performing TopK selection on raw logits and only computing
sigmoid for the selected TopK elements, reducing exp() operations from
num_experts to topk.
"""
import itertools

import pytest
import torch
from sgl_kernel import topk_sigmoid_opt


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_sigmoid_opt(num_tokens, num_experts, topk):
    """Test optimized kernel matches torch reference (no correction_bias)."""
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights,
        topk_indices,
        gating_output,
    )

    # Native torch implementation
    sigmoid_output = torch.sigmoid(gating_output)
    topk_weights_ref, topk_indices_ref = torch.topk(sigmoid_output, topk, dim=-1)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: torch={topk_weights_ref} vs SGLang={topk_weights}"

    assert torch.allclose(
        topk_indices_ref.int(), topk_indices, atol=0, rtol=0
    ), f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices}"


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, dtype",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
            [torch.float16, torch.bfloat16, torch.float32],  # dtype
        )
    ),
)
def test_topk_sigmoid_opt_dtype_regression(num_tokens, num_experts, topk, dtype):
    """Test optimized kernel with different input dtypes."""
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights,
        topk_indices,
        gating_output,
    )

    topk_weights_ref = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device="cuda"
    )
    topk_indices_ref = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights_ref,
        topk_indices_ref,
        gating_output.float(),
    )

    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: float32={topk_weights_ref} vs {dtype}={topk_weights}"

    assert torch.allclose(
        topk_indices_ref.int(), topk_indices, atol=0, rtol=0
    ), f"Indices mismatch: float32={topk_indices_ref}, {dtype}={topk_indices}"


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_sigmoid_opt_renormalize(num_tokens, num_experts, topk):
    """Test optimized kernel with renormalization."""
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=True,
    )

    topk_weights_ref = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device="cuda"
    )
    topk_indices_ref = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights_ref,
        topk_indices_ref,
        gating_output,
    )
    topk_weights_ref = topk_weights_ref / topk_weights_ref.sum(dim=-1, keepdim=True)

    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: w/o renormalize={topk_weights_ref} vs w/ renormalize={topk_weights}"

    assert torch.allclose(
        topk_indices_ref.int(), topk_indices, atol=0, rtol=0
    ), f"Indices mismatch: w/o renormalize={topk_indices_ref}, w/ renormalize={topk_indices}"


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048],  # num_tokens
            [4, 8, 16, 32, 48, 64, 128, 256],  # num_experts
            [1, 2, 4],  # topk
        )
    ),
)
def test_topk_sigmoid_opt_renormalize_correction_bias(num_tokens, num_experts, topk):
    """Test optimized kernel with correction_bias (falls back to original path)."""
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn((num_experts), dtype=torch.float32, device="cuda")

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    topk_sigmoid_opt(
        topk_weights,
        topk_indices,
        gating_output,
        renormalize=True,
        correction_bias=correction_bias,
    )

    # Native torch implementation
    sigmoid_output = torch.sigmoid(gating_output)
    sigmoid_scores = sigmoid_output.view(-1, num_experts) + correction_bias.unsqueeze(0)
    _, topk_indices_ref = torch.topk(sigmoid_scores, k=topk, dim=-1)
    topk_weights_ref = sigmoid_output.gather(1, topk_indices_ref)
    topk_weights_ref = topk_weights_ref / topk_weights_ref.sum(dim=-1, keepdim=True)

    # Verify the top-k weights and indices match the torch native ones
    assert torch.allclose(
        topk_weights_ref, topk_weights, atol=1e-3, rtol=1e-3
    ), f"Weights mismatch: torch={topk_weights_ref} vs SGLang={topk_weights}"

    assert torch.allclose(
        topk_indices_ref.int(), topk_indices, atol=0, rtol=0
    ), f"Indices mismatch: torch={topk_indices_ref}, SGLang={topk_indices}"


if __name__ == "__main__":
    pytest.main([__file__])
