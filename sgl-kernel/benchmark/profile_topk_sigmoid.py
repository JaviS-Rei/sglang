"""
Profile script for topk_sigmoid kernels with nsys.
Usage:
  nsys profile -o profile_original python profile_topk_sigmoid.py original
  nsys profile -o profile_opt python profile_topk_sigmoid.py opt
"""
import sys
import torch
from sgl_kernel import topk_sigmoid, topk_sigmoid_opt

# Configuration
NUM_TOKENS = 32768
NUM_EXPERTS = 256
TOPK = 8
WARMUP_ITERS = 10
PROFILE_ITERS = 100

def run_original():
    gating_output = torch.randn((NUM_TOKENS, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    topk_weights = torch.empty((NUM_TOKENS, TOPK), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((NUM_TOKENS, TOPK), dtype=torch.int32, device="cuda")
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        topk_sigmoid(topk_weights, topk_indices, gating_output, renormalize=True)
    torch.cuda.synchronize()
    
    # Profile
    for _ in range(PROFILE_ITERS):
        topk_sigmoid(topk_weights, topk_indices, gating_output, renormalize=True)
    torch.cuda.synchronize()

def run_opt():
    gating_output = torch.randn((NUM_TOKENS, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    topk_weights = torch.empty((NUM_TOKENS, TOPK), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((NUM_TOKENS, TOPK), dtype=torch.int32, device="cuda")
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        topk_sigmoid_opt(topk_weights, topk_indices, gating_output, renormalize=True)
    torch.cuda.synchronize()
    
    # Profile
    for _ in range(PROFILE_ITERS):
        topk_sigmoid_opt(topk_weights, topk_indices, gating_output, renormalize=True)
    torch.cuda.synchronize()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_topk_sigmoid.py [original|opt]")
        sys.exit(1)
    
    version = sys.argv[1]
    print(f"Profiling {version} version with:")
    print(f"  num_tokens={NUM_TOKENS}, num_experts={NUM_EXPERTS}, topk={TOPK}")
    print(f"  warmup_iters={WARMUP_ITERS}, profile_iters={PROFILE_ITERS}")
    
    if version == "original":
        run_original()
    elif version == "opt":
        run_opt()
    else:
        print(f"Unknown version: {version}")
        sys.exit(1)
    
    print("Done!")

