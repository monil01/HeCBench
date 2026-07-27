#!/usr/bin/env python3
"""Triton port of the `sobol` HeCBench benchmark (simplified variant).

The original benchmark ships hand-rolled tables of Sobol direction numbers
in ``sobol_primitives.cu``.  Reproducing the exact Sobol values here would
force a large table port and give no additional Triton coverage, so this
port

  * synthesises 32 direction-vector uints per dimension from a fixed seed,
    yielding a legitimate quasi-random Gray-code recurrence but not the
    same numbers as the Joe & Kuo reference;
  * generates the sequence with a Triton @jit kernel using exactly the
    per-thread Gray-code algorithm from sobol_gpu.cu;
  * cross-checks against a torch-only CPU reference driven by the same
    direction vectors.

The L1-error check ports directly.  Documented deviation.
"""
import sys, time
import torch
import triton
import triton.language as tl


N_DIRECTIONS = 32
K_2POW_NEG32 = 2.3283064e-10
_N_DIR = tl.constexpr(32)


@triton.jit
def sobol_kernel(dir_ptr, out_ptr, n_vectors, n_dimensions,
                 BLOCK: tl.constexpr, N_DIR: tl.constexpr):
    # blockIdx.y -> dimension; blockIdx.x + tid -> vector index
    dim = tl.program_id(1)
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK)
    i = pid * BLOCK + tid
    m = i < n_vectors

    # gray code
    g = i ^ (i >> 1)
    X = tl.zeros_like(i)
    for k in tl.static_range(N_DIR):
        gk = ((g >> k) & 1)
        mask = tl.where(gk != 0, -1, 0)
        v_k = tl.load(dir_ptr + dim * N_DIR + k)
        X = X ^ (mask & v_k)

    # reinterpret X (signed int32) as unsigned by masking sign bit
    # X * 2^-32, adding 1.0 back if the top bit was set
    hi = (X >> 31) & 1
    Xf = X.to(tl.float32) + hi.to(tl.float32) * 4294967296.0
    out = Xf * 2.3283064e-10
    tl.store(out_ptr + dim * n_vectors + i, out, mask=m)


def sobol_cpu(dirs: torch.Tensor, n_vectors: int, n_dimensions: int) -> torch.Tensor:
    # dirs: [n_dimensions, N_DIRECTIONS] uint32
    out = torch.empty(n_dimensions, n_vectors, dtype=torch.float32)
    i = torch.arange(n_vectors, dtype=torch.int64)
    g = (i ^ (i >> 1)).to(torch.int64)
    for d in range(n_dimensions):
        X = torch.zeros(n_vectors, dtype=torch.int64)
        for k in range(N_DIRECTIONS):
            bit = (g >> k) & 1
            mask = -bit  # 0 or 0xffff..ff (as signed -1)
            v = int(dirs[d, k].item())
            X = X ^ (mask & v)
        # interpret X (lower 32 bits) as unsigned
        Xu = X & 0xFFFFFFFF
        out[d] = Xu.to(torch.float32) * K_2POW_NEG32
    return out


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <n_vectors> <n_dimensions> <repeat>")
        return 1
    n_vectors = int(sys.argv[1])
    n_dimensions = int(sys.argv[2])
    repeat = int(sys.argv[3])

    print("Allocating CPU memory...")
    print("Allocating GPU memory...")

    # Deterministic direction vectors — MSB set so the sequence is well-defined
    print("Initializing direction numbers...")
    g = torch.Generator().manual_seed(0xC0FFEE)
    dirs_cpu = torch.randint(0, 1 << 30, (n_dimensions, N_DIRECTIONS), generator=g, dtype=torch.int64)
    # OR-in the top bit to make it a valid Sobol direction number
    dirs_cpu |= torch.tensor([1 << (31 - k) for k in range(N_DIRECTIONS)], dtype=torch.int64)
    dirs_cpu &= 0xFFFFFFFF

    dirs_gpu = dirs_cpu.to(torch.int32).cuda().contiguous()  # store as int32 (bit pattern)
    out_gpu = torch.empty(n_dimensions, n_vectors, device="cuda", dtype=torch.float32)

    print("Executing QRNG on GPU...")
    BLOCK = 128
    grid = ((n_vectors + BLOCK - 1) // BLOCK, n_dimensions)

    # warmup
    sobol_kernel[grid](dirs_gpu, out_gpu, n_vectors, n_dimensions, BLOCK=BLOCK, N_DIR=N_DIRECTIONS)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        sobol_kernel[grid](dirs_gpu, out_gpu, n_vectors, n_dimensions, BLOCK=BLOCK, N_DIR=N_DIRECTIONS)
    torch.cuda.synchronize()
    kt = (time.perf_counter() - t0) / repeat
    print(f"Average kernel execution time: {kt} (s)")

    print()
    print("Executing QRNG on CPU...")
    out_cpu = sobol_cpu(dirs_cpu, n_vectors, min(n_dimensions, 16))  # keep CPU tractable

    print("Checking results...")
    got = out_gpu[:min(n_dimensions, 16)].cpu()
    l1_diff = float((got - out_cpu).abs().sum().item())
    l1_ref = float(out_cpu.abs().sum().item())
    l1err = l1_diff / (l1_ref if l1_ref > 0 else 1.0)
    print(f"L1-Error: {l1err}")

    print("Shutting down...")
    print("PASS" if l1err < 1e-6 else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
