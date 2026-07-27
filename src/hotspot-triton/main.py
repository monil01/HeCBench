#!/usr/bin/env python3
"""Triton port of the `hotspot` HeCBench benchmark.

hotspot models transient heat diffusion on an integrated-circuit floorplan.
Every step is a 5-point stencil update:

    T'[i,j] = T + step/Cap * ( P
                             + (T_S + T_N - 2T) / Ry
                             + (T_E + T_W - 2T) / Rx
                             + (amb - T) / Rz )

Simplification notes:
  * The CUDA reference uses a pyramid tiling scheme (multiple stencil
    iterations per kernel launch via shared memory).  Numerically the
    result is identical to N separate one-step launches for a matching
    boundary, so this port fuses no iterations and launches one Triton
    kernel per iteration.
  * The reference input files (``../data/hotspot/temp_512`` and
    ``../data/hotspot/power_512``) are absent in this tree, so we
    synthesise deterministic inputs (uniform 273K + a hot patch, uniform
    power density with a localised hot power spot).  The CPU reference
    uses the same synthesised inputs, so the PASS check stays meaningful.
"""
import sys, os, math, time
import torch
import triton
import triton.language as tl


BLOCK_SIZE = 16

# Physical constants (from hotspot.h)
MAX_PD = 3.0e6
PRECISION = 0.001
SPEC_HEAT_SI = 1.75e6
K_SI = 100.0
FACTOR_CHIP = 0.5
t_chip = 0.0005
chip_height = 0.016
chip_width = 0.016
amb_temp = 80.0


@triton.jit
def calc_temp_kernel(power_ptr, src_ptr, dst_ptr,
                     rows, cols,
                     step_div_Cap, Rx_1, Ry_1, Rz_1,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    off_y = pid_y * BLOCK_M + tl.arange(0, BLOCK_M)
    off_x = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)
    y = off_y[:, None]
    x = off_x[None, :]

    in_range = (y >= 0) & (y < rows) & (x >= 0) & (x < cols)
    idx = y * cols + x

    # neighbour indices clamped to [0, rows/cols-1]
    ym = tl.where(y - 1 < 0, 0, y - 1)
    yp = tl.where(y + 1 > rows - 1, rows - 1, y + 1)
    xm = tl.where(x - 1 < 0, 0, x - 1)
    xp = tl.where(x + 1 > cols - 1, cols - 1, x + 1)

    T = tl.load(src_ptr + idx, mask=in_range, other=0.0)
    P = tl.load(power_ptr + idx, mask=in_range, other=0.0)
    Tn = tl.load(src_ptr + ym * cols + x, mask=in_range, other=0.0)
    Ts = tl.load(src_ptr + yp * cols + x, mask=in_range, other=0.0)
    Tw = tl.load(src_ptr + y * cols + xm, mask=in_range, other=0.0)
    Te = tl.load(src_ptr + y * cols + xp, mask=in_range, other=0.0)

    val = T + step_div_Cap * (P
                              + (Ts + Tn - 2.0 * T) * Ry_1
                              + (Te + Tw - 2.0 * T) * Rx_1
                              + (80.0 - T) * Rz_1)
    tl.store(dst_ptr + idx, val, mask=in_range)


def load_or_synth(path, rows, cols, default_val, hot_val):
    if os.path.exists(path):
        # ASCII, one value per line, rows*cols lines
        vals = []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                vals.append(float(s))
                if len(vals) == rows * cols:
                    break
        t = torch.tensor(vals, dtype=torch.float32).view(rows, cols)
        if t.numel() < rows * cols:
            raise RuntimeError(f"not enough values in {path}")
        return t
    # synthesize deterministic input
    t = torch.full((rows, cols), default_val, dtype=torch.float32)
    # place a hot patch in the middle
    r0, r1 = rows // 4, rows // 2
    c0, c1 = cols // 4, cols // 2
    t[r0:r1, c0:c1] = hot_val
    return t


def main():
    if len(sys.argv) < 7:
        print(f"Usage: {sys.argv[0]} <grid> <pyramid_height> <sim_time> <temp> <power> <output>")
        return 1

    grid_rows = int(sys.argv[1]); grid_cols = grid_rows
    pyramid_height = int(sys.argv[2])
    total_iterations = int(sys.argv[3])
    tfile, pfile, ofile = sys.argv[4], sys.argv[5], sys.argv[6]

    print(f"Work-group size of kernel = {BLOCK_SIZE} X {BLOCK_SIZE}")

    grid_h = chip_height / grid_rows
    grid_w = chip_width / grid_cols
    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_w * grid_h
    Rx = grid_w / (2.0 * K_SI * t_chip * grid_h)
    Ry = grid_h / (2.0 * K_SI * t_chip * grid_w)
    Rz = t_chip / (K_SI * grid_h * grid_w)
    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = PRECISION / max_slope

    temp0 = load_or_synth(tfile, grid_rows, grid_cols, 323.0, 400.0)
    power0 = load_or_synth(pfile, grid_rows, grid_cols, 0.05, 5.0)
    if not (os.path.exists(tfile) and os.path.exists(pfile)):
        print("(hotspot input files missing; using synthetic init data)")

    # --- CPU reference (torch-only, single-step iteration) ---
    ref_start = time.perf_counter()
    T = temp0.clone()
    for _ in range(total_iterations):
        T_up = torch.cat([T[:1], T[:-1]], dim=0)
        T_dn = torch.cat([T[1:], T[-1:]], dim=0)
        T_lf = torch.cat([T[:, :1], T[:, :-1]], dim=1)
        T_rt = torch.cat([T[:, 1:], T[:, -1:]], dim=1)
        T_new = T + (step / Cap) * (
            power0
            + (T_dn + T_up - 2.0 * T) / Ry
            + (T_rt + T_lf - 2.0 * T) / Rx
            + (amb_temp - T) / Rz
        )
        T = T_new
    ref_time = time.perf_counter() - ref_start
    print(f"Total reference execution time {ref_time} (s)")
    result_ref = T

    # --- GPU (Triton) ---
    dev_start = time.perf_counter()
    dP = power0.cuda().contiguous()
    dT0 = temp0.cuda().contiguous()
    dT1 = torch.empty_like(dT0)

    grid = ((grid_rows + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (grid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)

    torch.cuda.synchronize()
    kstart = time.perf_counter()
    src, dst = dT0, dT1
    for _ in range(total_iterations):
        calc_temp_kernel[grid](dP, src, dst,
                               grid_rows, grid_cols,
                               step / Cap, 1.0 / Rx, 1.0 / Ry, 1.0 / Rz,
                               BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE)
        src, dst = dst, src
    torch.cuda.synchronize()
    ktime = time.perf_counter() - kstart
    print(f"Total kernel execution time {ktime} (s)")

    got = src.cpu()
    dev_time = time.perf_counter() - dev_start
    print(f"Device offloading time: {dev_time:.3f} (s)")

    err = (got - result_ref).abs().max().item()
    print("PASS" if err < 1e-3 else "FAIL")

    # write output
    try:
        with open(ofile, 'w') as f:
            flat = got.view(-1)
            for i, v in enumerate(flat.tolist()):
                f.write(f"{i}\t{v:g}\n")
    except Exception as e:
        print(f"(could not write {ofile}: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
