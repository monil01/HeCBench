#!/usr/bin/env python3
"""Triton port of the `haversine` HeCBench benchmark.

Computes great-circle distances between 2**21 city locations and 6 reference
cities.  The CUDA driver reads city coords from ``locations.txt`` -- that
symlink is broken in the current tree (haversine-cuda/locations.txt ->
../geodesic-sycl/locations.txt is dangling), so this port falls back to
deterministic synthetic lat/lon coordinates when the file is missing.
"""
import sys, os, math, time
import torch
import triton
import triton.language as tl


EARTH_RADIUS_KM = 6371.0
DEG2RAD = math.pi / 180.0


@triton.jit
def haversine_kernel(ax_ptr, ay_ptr, bx_ptr, by_ptr, dist_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    ax = tl.load(ax_ptr + offs, mask=m, other=0.0)
    ay = tl.load(ay_ptr + offs, mask=m, other=0.0)
    bx = tl.load(bx_ptr + offs, mask=m, other=0.0)
    by = tl.load(by_ptr + offs, mask=m, other=0.0)

    x = (bx - ax) * 0.5
    y = (by - ay) * 0.5
    sy = tl.sin(y)
    sx = tl.sin(x)
    scale = tl.cos(ay) * tl.cos(by)
    inner = sy * sy + sx * sx * scale
    # asin(x) = atan2(x, sqrt(1-x*x))
    root = tl.sqrt(inner)
    d = 2.0 * 6371.0 * tl.extra.libdevice.asin(root)
    tl.store(dist_ptr + offs, d, mask=m)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file> <repeat>")
        return 1
    filename = sys.argv[1]
    repeat = int(sys.argv[2])

    num_cities = 1 << 21
    num_ref = 6
    index_map = [436483, 1952407, 627919, 377884, 442703, 1863423]
    N = num_cities * num_ref

    print(f"Reading city locations from file {filename}...")
    lat = torch.empty(num_cities, dtype=torch.float64)
    lon = torch.empty(num_cities, dtype=torch.float64)
    read_from_file = os.path.exists(filename)
    if read_from_file:
        with open(filename) as f:
            i = 0
            for line in f:
                p = line.split()
                if len(p) < 2:
                    continue
                lat[i] = float(p[0])
                lon[i] = float(p[1])
                i += 1
                if i == num_cities:
                    break
        if i < num_cities:
            # pad with synthetic data
            g = torch.Generator().manual_seed(0)
            lat[i:] = (torch.rand(num_cities - i, generator=g, dtype=torch.float64) - 0.5) * 180.0
            lon[i:] = (torch.rand(num_cities - i, generator=g, dtype=torch.float64) - 0.5) * 360.0
    else:
        # locations.txt missing -> synthesize
        print(f"(file {filename} missing; using deterministic synthetic locations)")
        g = torch.Generator().manual_seed(0)
        lat = (torch.rand(num_cities, generator=g, dtype=torch.float64) - 0.5) * 180.0
        lon = (torch.rand(num_cities, generator=g, dtype=torch.float64) - 0.5) * 360.0

    # Build the 6*num_cities pairs (a_lat, a_lon, b_lat, b_lon).
    # convert to radians up front to keep the kernel simple
    ay_all = (lat * DEG2RAD).repeat(num_ref).cuda()  # a_lat as radians
    ax_all = (lon * DEG2RAD).repeat(num_ref).cuda()

    by_all = torch.empty(N, dtype=torch.float64)
    bx_all = torch.empty(N, dtype=torch.float64)
    for c, ref_idx in enumerate(index_map):
        ref = ref_idx - 1
        by_all[c*num_cities:(c+1)*num_cities] = lat[ref] * DEG2RAD
        bx_all[c*num_cities:(c+1)*num_cities] = lon[ref] * DEG2RAD
    by_all = by_all.cuda()
    bx_all = bx_all.cuda()

    dist = torch.empty(N, device="cuda", dtype=torch.float64)

    # CPU reference (torch on CPU tensors)
    lat_cpu = lat.repeat(num_ref) * DEG2RAD
    lon_cpu = lon.repeat(num_ref) * DEG2RAD
    by_cpu = by_all.cpu()
    bx_cpu = bx_all.cpu()
    x = (bx_cpu - lon_cpu) * 0.5
    y = (by_cpu - lat_cpu) * 0.5
    inner = y.sin() ** 2 + x.sin() ** 2 * lat_cpu.cos() * by_cpu.cos()
    ref_dist = 2.0 * EARTH_RADIUS_KM * inner.sqrt().asin()

    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)

    # warmup
    haversine_kernel[grid](ax_all, ay_all, bx_all, by_all, dist, N, BLOCK=BLOCK)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        haversine_kernel[grid](ax_all, ay_all, bx_all, by_all, dist, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    kt_ms = (time.perf_counter() - t0) * 1000.0 / repeat
    print(f"Average execution time of haversine kernel: {kt_ms:f} (ms)")

    got = dist.cpu()
    err = float((got - ref_dist).abs().max().item())
    print(f"The maximum error in distance is {err:f}")
    print("PASS" if err < 1e-6 else "FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
