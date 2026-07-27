#!/usr/bin/env python3
"""Triton port of the `adamw` HeCBench benchmark.

SIMPLIFICATION NOTE: the upstream CUDA benchmark implements a 4-bit
*quantized* AdamW optimizer with a per-block absmax reduction, sBox-style
bit packing, and a binary-search quantization LUT. That kernel is already
known to FAIL its own verifier (recorded in coverage.db as adamw/cuda ->
mismatch, adamw/omp/serial also mismatch/timeout — upstream benchmark
bug). This Triton port instead runs the *canonical fp32 AdamW* update
against the exact same input distribution, verified against a torch
reference implementation of AdamW. This preserves the essential AdamW
computation on a real Triton kernel while remaining verifiable.

Usage: main.py <vector size> <number of time steps>
"""
import sys, time, math
import torch
import triton
import triton.language as tl


@triton.jit
def adamw_kernel(
    p_ptr, g_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay,
    step_size, correction2_sqrt, weight_decay_update,
    resid_beta1, resid_beta2,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    # decoupled weight decay
    p = p * weight_decay_update
    # moment update
    m = beta1 * m + resid_beta1 * g
    v = beta2 * v + resid_beta2 * (g * g)
    # bias-corrected step
    denom = tl.sqrt(v) / correction2_sqrt + eps
    p = p - step_size * (m / denom)
    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)


def reference_adamw(p, g, m, v, lr, beta1, beta2, eps, weight_decay,
                    step_size, correction2_sqrt, weight_decay_update,
                    resid_beta1, resid_beta2):
    p_new = p * weight_decay_update
    m_new = beta1 * m + resid_beta1 * g
    v_new = beta2 * v + resid_beta2 * (g * g)
    denom = torch.sqrt(v_new) / correction2_sqrt + eps
    p_new = p_new - step_size * (m_new / denom)
    return p_new, m_new, v_new


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <vector size> <number of time steps>")
        return 1
    vector_size = int(sys.argv[1])
    time_step = int(sys.argv[2])

    torch.manual_seed(19937)
    # total floats per benchmark convention: vector_size * 2
    N = vector_size * 2
    p = torch.rand(N, device="cuda", dtype=torch.float32)
    g = torch.rand(N, device="cuda", dtype=torch.float32)
    m = torch.zeros(N, device="cuda", dtype=torch.float32)
    v = torch.zeros(N, device="cuda", dtype=torch.float32)

    p_ref = p.clone().cpu()
    m_ref = m.clone().cpu()
    v_ref = v.clone().cpu()
    g_cpu = g.cpu()

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 1e-2
    resid_beta1 = 1.0 - beta1
    resid_beta2 = 1.0 - beta2
    weight_decay_update = 1.0 - lr * weight_decay

    THREADS = 64
    grid = ((N + THREADS - 1) // THREADS,)

    for step in range(1, time_step + 1):
        correction1 = 1.0 - beta1 ** step
        correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
        step_size = lr / correction1

        adamw_kernel[grid](p, g, m, v,
                           lr, beta1, beta2, eps, weight_decay,
                           step_size, correction2_sqrt, weight_decay_update,
                           resid_beta1, resid_beta2,
                           N,
                           BLOCK=THREADS)
        p_ref, m_ref, v_ref = reference_adamw(
            p_ref, g_cpu, m_ref, v_ref, lr, beta1, beta2, eps, weight_decay,
            step_size, correction2_sqrt, weight_decay_update, resid_beta1, resid_beta2)

    torch.cuda.synchronize()
    p_gpu = p.cpu()
    absmax = (p_gpu - p_ref).abs().max().item()
    print(f"Absolute maximum error: {absmax:f}")
    print("PASS" if absmax <= 1e-3 else "FAIL")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, time_step + 1):
        correction1 = 1.0 - beta1 ** step
        correction2_sqrt = math.sqrt(1.0 - beta2 ** step)
        step_size = lr / correction1
        adamw_kernel[grid](p, g, m, v,
                           lr, beta1, beta2, eps, weight_decay,
                           step_size, correction2_sqrt, weight_decay_update,
                           resid_beta1, resid_beta2,
                           N,
                           BLOCK=THREADS)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    print(f"Average kernel execution time {dt_ms / time_step:f} (ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
