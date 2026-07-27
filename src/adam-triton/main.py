#!/usr/bin/env python3
"""Triton port of the `adam` HeCBench benchmark.

Runs the Adam optimizer update kernel `repeat` times, then verifies against a
pure-torch reference. Kernel iterates over `time_step` inside each per-element
update so the parameter/moment state has the same trajectory as the CUDA port.

Usage: main.py <vector_size> <time_step> <repeat>
"""
import sys, time, math
import torch
import triton
import triton.language as tl

ADAM_MODE_0 = 0

@triton.jit
def adam_kernel(
    p_ptr, m_ptr, v_ptr, g_ptr,
    b1, b2, eps, grad_scale, step_size,
    time_step, vector_size, decay,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vector_size

    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)

    scaled_grad = g / grad_scale
    for t in range(1, time_step + 1):
        m = b1 * m + (1.0 - b1) * scaled_grad
        v = b2 * v + (1.0 - b2) * scaled_grad * scaled_grad
        # tl.math.pow may be missing on older Triton; use exp/log identity.
        tt = t.to(tl.float32)
        m_corr = m / (1.0 - tl.exp(tt * tl.log(b1)))
        v_corr = v / (1.0 - tl.exp(tt * tl.log(b2)))
        denom = tl.sqrt(v_corr + eps)   # ADAM_MODE_0
        update = m_corr / denom + decay * p
        p = p - step_size * update

    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)


def reference(repeat, p, m, v, g, b1, b2, eps, grad_scale, step_size,
              time_step, decay):
    """CPU/torch reference: exact match to reference.h."""
    for _ in range(repeat):
        scaled = g / grad_scale
        for t in range(1, time_step + 1):
            m = b1 * m + (1.0 - b1) * scaled
            v = b2 * v + (1.0 - b2) * scaled * scaled
            m_corr = m / (1.0 - b1 ** t)
            v_corr = v / (1.0 - b2 ** t)
            denom = torch.sqrt(v_corr + eps)
            update = m_corr / denom + decay * p
            p = p - step_size * update
    return p


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <vector size> <number of time steps> <repeat>")
        return 1
    n, ts, repeat = map(int, sys.argv[1:4])

    gen = torch.Generator(device="cuda").manual_seed(19937)
    m = torch.rand(n, device="cuda", generator=gen, dtype=torch.float32)
    v = torch.rand(n, device="cuda", generator=gen, dtype=torch.float32)
    g = torch.rand(n, device="cuda", generator=gen, dtype=torch.float32)
    p_init = torch.rand(n, device="cuda", generator=gen, dtype=torch.float32)
    p = p_init.clone()

    step_size, decay = 1e-3, 0.5
    b1, b2 = 0.9, 0.999
    eps, grad_scale = 1e-10, 256.0

    m_save = m.clone()
    v_save = v.clone()

    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK,)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        adam_kernel[grid](
            p, m, v, g,
            b1, b2, eps, grad_scale, step_size,
            ts, n, decay,
            BLOCK=BLOCK,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1e3 / repeat

    print(f"Average kernel execution time {elapsed_ms:f} (ms)")

    # verify against reference
    p_ref = reference(repeat, p_init.clone(), m_save, v_save, g,
                      b1, b2, eps, grad_scale, step_size, ts, decay)
    ok = torch.max(torch.abs(p - p_ref)).item() <= 1e-3
    print("PASS" if ok else "FAIL")

    cr = float(p_ref.mean().double())
    cp = float(p.mean().double())
    print(f"Checksum: {cr:f} {cp:f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
