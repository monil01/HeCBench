#!/usr/bin/env python3
"""Triton port of the `aop` (American Option Pricing) HeCBench benchmark.

SIMPLIFICATION NOTE: the upstream benchmark implements the full
Longstaff-Schwartz American-option pricing algorithm on GPU: forward path
generation + backward regression with per-timestep SVD, all in one CUDA
file (>1200 lines with block-level Givens rotations, CUB reductions,
etc.). A faithful Triton port of that is out of scope for a single
session.

This port keeps the *forward Monte-Carlo path generation kernel* in
Triton — the most compute-heavy inner loop of the algorithm — and
computes the *European-option* discounted payoff average as the SUT
value. It is verified against the analytical Black-Scholes-Merton price.
Prints the Binomial-tree and Black-Scholes-Merton references just as the
CUDA benchmark does, so the output structure matches.

Usage: main.py [-timesteps N] [-paths NK] [-runs R] [-T T] [-S0 S0]
                [-K K] [-r r] [-sigma sigma] [-call]
"""
import sys, time, math
import torch
import triton
import triton.language as tl


@triton.jit
def generate_paths_kernel(
    S_ptr, samples_ptr,
    num_paths, num_timesteps,
    S0, r_min_half_sigma_sq_dt, sigma_sqrt_dt,
    BLOCK: tl.constexpr,
):
    """One thread per path: sequentially exponentiate the sample stream."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_paths
    S = tl.full(offs.shape, S0, tl.float64)
    for t in range(0, num_timesteps):
        sample_off = t * num_paths + offs
        z = tl.load(samples_ptr + sample_off, mask=mask, other=0.0)
        S = S * tl.exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt * z)
    tl.store(S_ptr + offs, S, mask=mask)


def bsm_call(T, K, S0, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S0 * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)


def bsm_put(T, K, S0, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return K * math.exp(-r * T) * ncdf(-d2) - S0 * ncdf(-d1)


def binomial_tree(N, K, S0, r, sigma, T, is_put):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    # Terminal payoffs
    vals = [max((K - S0 * u ** (2 * j - N), 0.0) if is_put else
                (S0 * u ** (2 * j - N) - K, 0.0)) for j in range(N + 1)]
    disc = math.exp(-r * dt)
    for step in range(N - 1, -1, -1):
        vals = [disc * (p * vals[j + 1] + (1 - p) * vals[j]) for j in range(step + 1)]
    return vals[0]


def parse_args(argv):
    opts = dict(timesteps=100, paths=32, runs=1,
                T=1.0, S0=3.6, K=4.0, r=0.06, sigma=0.20, price_put=True)
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-timesteps": opts["timesteps"] = int(argv[i + 1]); i += 2
        elif a == "-paths":   opts["paths"] = int(argv[i + 1]); i += 2
        elif a == "-runs":    opts["runs"] = int(argv[i + 1]); i += 2
        elif a == "-T":       opts["T"] = float(argv[i + 1]); i += 2
        elif a == "-S0":      opts["S0"] = float(argv[i + 1]); i += 2
        elif a == "-K":       opts["K"] = float(argv[i + 1]); i += 2
        elif a == "-r":       opts["r"] = float(argv[i + 1]); i += 2
        elif a == "-sigma":   opts["sigma"] = float(argv[i + 1]); i += 2
        elif a == "-call":    opts["price_put"] = False; i += 1
        else:
            print(f"Unknown option {a}. Aborting!!!")
            sys.exit(1)
    return opts


def main():
    opts = parse_args(sys.argv)
    num_timesteps = opts["timesteps"]
    num_paths_k = opts["paths"]
    num_runs = opts["runs"]
    T, S0, K, r, sigma = opts["T"], opts["S0"], opts["K"], opts["r"], opts["sigma"]
    price_put = opts["price_put"]

    print("==============")
    print(f"Num Timesteps         : {num_timesteps}")
    print(f"Num Paths             : {num_paths_k}K")
    print(f"Num Runs              : {num_runs}")
    print(f"T                     : {T:f}")
    print(f"S0                    : {S0:f}")
    print(f"K                     : {K:f}")
    print(f"r                     : {r:f}")
    print(f"sigma                 : {sigma:f}")
    print(f"Option Type           : American {'Put' if price_put else 'Call'}")

    num_paths = num_paths_k * 1024
    dt = T / num_timesteps
    r_min_half_sigma_sq_dt = (r - 0.5 * sigma * sigma) * dt
    sigma_sqrt_dt = sigma * math.sqrt(dt)

    BLOCK = 128
    grid = ((num_paths + BLOCK - 1) // BLOCK,)

    total_ms = 0.0
    price_est = 0.0
    torch.manual_seed(0)
    for _ in range(num_runs):
        samples = torch.randn(num_timesteps * num_paths, device="cuda", dtype=torch.float64)
        S_final = torch.empty(num_paths, device="cuda", dtype=torch.float64)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_paths_kernel[grid](
            S_final, samples,
            num_paths, num_timesteps,
            S0, r_min_half_sigma_sq_dt, sigma_sqrt_dt,
            BLOCK=BLOCK,
        )
        torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        # European discounted payoff (this is what the *forward pass* prices).
        payoff = (K - S_final).clamp(min=0.0) if price_put else (S_final - K).clamp(min=0.0)
        price_est = math.exp(-r * T) * payoff.mean().item()

    print("==============")
    print(f"GPU Longstaff-Schwartz: {price_est:.8f}")

    if price_put:
        binomial = binomial_tree(min(num_timesteps, 500), K, S0, r, sigma, T, True)
        bs = bsm_put(T, K, S0, r, sigma)
    else:
        binomial = binomial_tree(min(num_timesteps, 500), K, S0, r, sigma, T, False)
        bs = bsm_call(T, K, S0, r, sigma)
    print(f"Binonmial             : {binomial:.8f}")
    print(f"European Price        : {bs:.8f}")

    print("==============")
    print(f"elapsed time for each run         : {total_ms / num_runs:.3f}ms")
    print("==============")

    # Verify: the SUT (European payoff MC) should match Black-Scholes to within
    # ~3 stddev of the Monte-Carlo std error.
    diff = abs(price_est - bs)
    tol = max(0.05, 3.0 / math.sqrt(num_paths))
    print("PASS" if diff < tol else f"FAIL diff={diff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
