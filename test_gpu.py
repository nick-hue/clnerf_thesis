#!/usr/bin/env python3
import os
import time
import torch

def main():
    # 1) What devices does CUDA see?
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    print(f"CUDA_VISIBLE_DEVICES = {cuda_vis!r}")
    
    # 2) How many GPUs does torch see?
    n_gpus = torch.cuda.device_count()
    print(f"torch.cuda.device_count() = {n_gpus}")
    if n_gpus == 0:
        print("No GPUs detected. Exiting.")
        return

    # 3) Print the name of GPU 0 (which maps to the only visible CUDA device)
    print(f"Using device 0 → {torch.cuda.get_device_name(0)}\n")

    # 4) Do a quick matrix multiply on the GPU
    device = torch.device("cuda:0")
    print("Allocating two 10k×10k tensors on GPU...")
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    torch.cuda.synchronize()

    print("Running torch.mm(a, b) on the GPU...")
    t0 = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"Matrix multiply completed in {elapsed:.3f} seconds on {device}.")

if __name__ == "__main__":
    main()
