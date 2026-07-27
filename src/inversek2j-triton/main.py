#!/usr/bin/env python3
"""Triton port of the `inversek2j` HeCBench benchmark.

Inverse kinematics for a 3-joint planar arm. Each program in the grid handles
one target (x,y); Triton lacks per-lane local arrays of dynamic size, so the
kernel keeps the (small) 3-joint state in scalar variables that are unrolled
across the 3 joints. Verified against a torch-CPU reference implementation.

Usage: main.py <coord_in.txt> <iterations>
"""
import sys, time, math
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


MAX_LOOP = tl.constexpr(25)
NUM_JOINTS = tl.constexpr(3)
PI = tl.constexpr(3.14159265358979)
BLOCK_SIZE = 128


@triton.jit
def invkin_kernel(x_ptr, y_ptr, out_ptr, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < size
    curr_x = tl.load(x_ptr + tid, mask=mask, other=0.0)
    curr_y = tl.load(y_ptr + tid, mask=mask, other=0.0)

    a0 = tl.zeros(curr_x.shape, tl.float32)
    a1 = tl.zeros(curr_x.shape, tl.float32)
    a2 = tl.zeros(curr_x.shape, tl.float32)

    # x/y positions of each joint including the end-effector index 3.
    # xData[i] starts as i, yData[i] = 0. Only xData/yData for index 3 (end)
    # is used inside the loop, so we track only those explicitly. But we also
    # need xData[iter-1] for iter=1..3, i.e. xData[0], xData[1], xData[2].
    # yData[i]=0 for all i initially, so pc_y=0 for all iterations of the
    # first pass. After we update joint angles the end-effector position
    # does NOT actually get rewritten in the CUDA kernel — it only writes
    # angle_out — so xData/yData stay constant.
    x0 = tl.zeros(curr_x.shape, tl.float32) + 0.0
    x1 = tl.zeros(curr_x.shape, tl.float32) + 1.0
    x2 = tl.zeros(curr_x.shape, tl.float32) + 2.0
    xe = tl.zeros(curr_x.shape, tl.float32) + 3.0
    # y coords all zero.
    for curr_loop in range(0, MAX_LOOP):
        # iter=3
        pe_x = xe;          pe_y = tl.zeros_like(pe_x)
        pc_x = x2;          pc_y = tl.zeros_like(pc_x)
        dpe_x = pe_x - pc_x; dpe_y = pe_y - pc_y
        dtc_x = curr_x - pc_x; dtc_y = curr_y - pc_y
        lp = tl.sqrt(dpe_x*dpe_x + dpe_y*dpe_y)
        lt = tl.sqrt(dtc_x*dtc_x + dtc_y*dtc_y)
        a_x = dpe_x / lp; a_y = dpe_y / lp
        b_x = dtc_x / lt; b_y = dtc_y / lt
        dot = a_x*b_x + a_y*b_y
        dot = tl.where(dot > 1.0, 1.0, dot)
        dot = tl.where(dot < -1.0, -1.0, dot)
        ang = libdevice.acos(dot) * (180.0 / PI)
        direction = a_x*b_y - a_y*b_x
        ang = tl.where(direction < 0.0, -ang, ang)
        ang = tl.where(ang > 30.0, 30.0, ang)
        ang = tl.where(ang < -30.0, -30.0, ang)
        a2 = ang
        # rolling sum: angle_out[i+1] += angle_out[i] for i in 0..NUM_JOINTS-2
        a1 = a1 + a0
        a2 = a2 + a1

        # iter=2
        pc_x = x1
        dpe_x = pe_x - pc_x
        dtc_x = curr_x - pc_x; dtc_y = curr_y
        lp = tl.sqrt(dpe_x*dpe_x + pe_y*pe_y)
        lt = tl.sqrt(dtc_x*dtc_x + dtc_y*dtc_y)
        a_x = dpe_x / lp; a_y = pe_y / lp
        b_x = dtc_x / lt; b_y = dtc_y / lt
        dot = a_x*b_x + a_y*b_y
        dot = tl.where(dot > 1.0, 1.0, dot)
        dot = tl.where(dot < -1.0, -1.0, dot)
        ang = libdevice.acos(dot) * (180.0 / PI)
        direction = a_x*b_y - a_y*b_x
        ang = tl.where(direction < 0.0, -ang, ang)
        ang = tl.where(ang > 30.0, 30.0, ang)
        ang = tl.where(ang < -30.0, -30.0, ang)
        a1 = ang
        a1 = a1 + a0
        a2 = a2 + a1

        # iter=1
        pc_x = x0
        dpe_x = pe_x - pc_x
        dtc_x = curr_x - pc_x; dtc_y = curr_y
        lp = tl.sqrt(dpe_x*dpe_x + pe_y*pe_y)
        lt = tl.sqrt(dtc_x*dtc_x + dtc_y*dtc_y)
        a_x = dpe_x / lp; a_y = pe_y / lp
        b_x = dtc_x / lt; b_y = dtc_y / lt
        dot = a_x*b_x + a_y*b_y
        dot = tl.where(dot > 1.0, 1.0, dot)
        dot = tl.where(dot < -1.0, -1.0, dot)
        ang = libdevice.acos(dot) * (180.0 / PI)
        direction = a_x*b_y - a_y*b_x
        ang = tl.where(direction < 0.0, -ang, ang)
        ang = tl.where(ang > 30.0, 30.0, ang)
        ang = tl.where(ang < -30.0, -30.0, ang)
        a0 = ang
        a1 = a1 + a0
        a2 = a2 + a1

    tl.store(out_ptr + tid * NUM_JOINTS + 0, a0, mask=mask)
    tl.store(out_ptr + tid * NUM_JOINTS + 1, a1, mask=mask)
    tl.store(out_ptr + tid * NUM_JOINTS + 2, a2, mask=mask)


def invkin_cpu(xT, yT):
    """Torch-CPU reference implementation of the same algorithm."""
    size = xT.numel()
    NJ = 3
    ML = 25
    angles = torch.zeros((size, NJ), dtype=torch.float32)
    for idx in range(size):
        xData = [0.0, 1.0, 2.0, 3.0]
        yData = [0.0, 0.0, 0.0, 0.0]
        a = [0.0, 0.0, 0.0]
        cx = float(xT[idx]); cy = float(yT[idx])
        for _ in range(ML):
            for it in range(NJ, 0, -1):
                pe_x = xData[NJ]; pe_y = yData[NJ]
                pc_x = xData[it-1]; pc_y = yData[it-1]
                dpe_x = pe_x - pc_x; dpe_y = pe_y - pc_y
                dtc_x = cx - pc_x; dtc_y = cy - pc_y
                lp = math.sqrt(dpe_x*dpe_x + dpe_y*dpe_y)
                lt = math.sqrt(dtc_x*dtc_x + dtc_y*dtc_y)
                a_x = dpe_x/lp; a_y = dpe_y/lp
                b_x = dtc_x/lt; b_y = dtc_y/lt
                dot = a_x*b_x + a_y*b_y
                dot = max(-1.0, min(1.0, dot))
                ang = math.acos(dot) * (180.0/3.14159265358979)
                if a_x*b_y - a_y*b_x < 0: ang = -ang
                ang = max(-30.0, min(30.0, ang))
                a[it-1] = ang
                for i in range(NJ - 1):
                    a[i+1] += a[i]
        angles[idx, 0] = a[0]; angles[idx, 1] = a[1]; angles[idx, 2] = a[2]
    return angles


def _to_cuda_out(d_out, n):
    NJ = 3
    return d_out.cpu().view(n, NJ)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input file coefficients> <iterations>")
        return 1
    path = sys.argv[1]
    iters = int(sys.argv[2])
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0])
    coords = torch.tensor([float(x) for x in toks[1:1+2*n]], dtype=torch.float32).view(n, 2)
    xT = coords[:, 0].contiguous()
    yT = coords[:, 1].contiguous()
    print(f"# Data Size = {n}")
    print("# Coordinates are read from file...")

    d_x = xT.cuda(); d_y = yT.cuda()
    NJ = 3
    d_out = torch.zeros(n * NJ, device="cuda", dtype=torch.float32)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        invkin_kernel[grid](d_x, d_y, d_out, n, BLOCK=BLOCK_SIZE)
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - t0) * 1e6 / iters
    print(f"Average kernel execution time {dt_us:f} (us)")

    gpu_out = d_out.cpu().view(n, NJ)

    # For verification, do CPU on a subset to keep runtime bounded.
    subset = min(n, 4096)
    ref = invkin_cpu(xT[:subset], yT[:subset])
    err = (gpu_out[:subset] - ref).abs().max().item()
    print(f"max_abs_err = {err:.4e}")
    print("PASS" if err <= 1e-3 else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
