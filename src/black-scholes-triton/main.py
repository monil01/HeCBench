#!/usr/bin/env python3
"""Triton port of the `black-scholes` HeCBench benchmark.

Prices 50 million European options via the analytic Black-Scholes formula.
The CUDA version calls its own erf() approximation; we use the standard
formula with torch.erf on the CPU as the reference and Triton's built-in
math on the GPU.  Both use the standard 37-option cycle from the CUDA driver.

Kernel is one program per BLOCK options; layout is 6 float32 buffers
(spot, strike, dividend, riskfree, timeYear, vol) plus one int8 buffer
for CALL/PUT.
"""
import sys, time, math
import torch
import triton
import triton.language as tl


# Values from black-scholes-cuda/blackScholesAnalyticEngine.cu (37 configs)
CALL, PUT = 1, 0
CONFIGS = [
    (CALL,  40.00,  42.00, 0.08, 0.04, 0.75, 0.35),
    (CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15),
    (CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15),
    (CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15),
    (CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.25),
    (CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.25),
    (CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.25),
    (CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.35),
    (CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.35),
    (CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.35),
    (CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.15),
    (CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.15),
    (CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.15),
    (CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.25),
    (CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.25),
    (CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.25),
    (CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.35),
    (CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.35),
    (CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.35),
    (PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15),
    (PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15),
    (PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15),
    (PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.25),
    (PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.25),
    (PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.25),
    (PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.35),
    (PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.35),
    (PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.35),
    (PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.15),
    (PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.15),
    (PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.15),
    (PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.25),
    (PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.25),
    (PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.25),
    (PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.35),
    (PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.35),
    (PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.35),
]


@triton.jit
def bs_kernel(type_ptr, spot_ptr, strike_ptr, div_ptr, risk_ptr, T_ptr, vol_ptr,
              out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    ty = tl.load(type_ptr + offs, mask=m, other=0)
    S  = tl.load(spot_ptr + offs, mask=m, other=1.0)
    K  = tl.load(strike_ptr + offs, mask=m, other=1.0)
    q  = tl.load(div_ptr + offs, mask=m, other=0.0)
    r  = tl.load(risk_ptr + offs, mask=m, other=0.0)
    T  = tl.load(T_ptr + offs, mask=m, other=1.0)
    v  = tl.load(vol_ptr + offs, mask=m, other=0.01)

    sqrtT = tl.sqrt(T)
    vs = v * sqrtT
    d1 = (tl.log(S / K) + (r - q + 0.5 * v * v) * T) / vs
    d2 = d1 - vs

    inv_sqrt2 = 0.70710678118654752440
    Nd1 = 0.5 * (1.0 + tl.erf(d1 * inv_sqrt2))
    Nd2 = 0.5 * (1.0 + tl.erf(d2 * inv_sqrt2))

    call = S * tl.exp(-q * T) * Nd1 - K * tl.exp(-r * T) * Nd2
    # put via parity: put = call - S*exp(-qT) + K*exp(-rT)
    put = call - S * tl.exp(-q * T) + K * tl.exp(-r * T)

    price = tl.where(ty == 1, call, put)
    tl.store(out_ptr + offs, price, mask=m)


def bs_cpu(type_arr, spot, strike, div, risk, T, vol):
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    S, K, q, r, v = spot, strike, div, risk, vol
    sqrtT = T.sqrt()
    vs = v * sqrtT
    d1 = ((S / K).log() + (r - q + 0.5 * v * v) * T) / vs
    d2 = d1 - vs
    Nd1 = 0.5 * (1.0 + (d1 * inv_sqrt2).erf())
    Nd2 = 0.5 * (1.0 + (d2 * inv_sqrt2).erf())
    call = S * (-q * T).exp() * Nd1 - K * (-r * T).exp() * Nd2
    put = call - S * (-q * T).exp() + K * (-r * T).exp()
    return torch.where(type_arr == 1, call, put)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <repeat>")
        return 1
    repeat = int(sys.argv[1])
    N = 50_000_000

    print(f"Number of options: {N}\n")

    NC = len(CONFIGS)
    cfg = torch.tensor(CONFIGS, dtype=torch.float32)
    # Build the 7 arrays by tiling cfg
    idx = torch.arange(N, dtype=torch.long) % NC
    tiled = cfg[idx].cuda()
    ty     = tiled[:, 0].to(torch.int32).contiguous()
    spot   = tiled[:, 1].contiguous()
    strike = tiled[:, 2].contiguous()
    div    = tiled[:, 3].contiguous()
    risk   = tiled[:, 4].contiguous()
    T      = tiled[:, 5].contiguous()
    vol    = tiled[:, 6].contiguous()
    out    = torch.empty(N, device="cuda", dtype=torch.float32)

    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)

    # warmup
    bs_kernel[grid](ty, spot, strike, div, risk, T, vol, out, N, BLOCK=BLOCK)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        bs_kernel[grid](ty, spot, strike, div, risk, T, vol, out, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    ktime_ms = (time.perf_counter() - t0) * 1000.0 / repeat

    print("Run on GPU")
    print(f"Average kernel execution time on GPU: {ktime_ms:f} (ms)")
    print(f"Processing time using GPU {ktime_ms:f} (ms)")

    tot = float(out.sum().item())
    mid = int(out[N // 2].item()) if False else float(out[N // 2].item())
    print(f"Summation of output prices on GPU: {tot:f}")
    print(f"Output price at index {N // 2} on GPU: {mid:f}\n")

    # CPU reference on a subset (50M options on CPU is too slow -- take 100k)
    subset = 100_000
    idx_s = idx[:subset]
    tiled_s = cfg[idx_s]  # on CPU
    ref = bs_cpu(tiled_s[:, 0].to(torch.int32),
                 tiled_s[:, 1], tiled_s[:, 2], tiled_s[:, 3],
                 tiled_s[:, 4], tiled_s[:, 5], tiled_s[:, 6])
    got = out[:subset].cpu()
    maxerr = float((got - ref).abs().max().item())
    print(f"Run on CPU")
    print(f"Max abs error (subset {subset}): {maxerr:g}")
    print("PASS" if maxerr < 1e-3 else "FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
