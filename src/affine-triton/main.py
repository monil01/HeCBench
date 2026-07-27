#!/usr/bin/env python3
"""Triton port of the `affine` HeCBench benchmark.

Applies an inverse-affine + bilinear-interpolation warp to a 512x512
uint16 grayscale image. Mirrors affine-cuda/kernel.h.

Usage: main.py <input.raw> <output.raw> <iterations>
"""
import sys, os, time, math, struct
import numpy as np
import torch
import triton
import triton.language as tl

X_SIZE = 512
Y_SIZE = 512
WHITE = 1


@triton.jit
def affine_kernel(
    src_ptr, dst_ptr,
    X_SIZE: tl.constexpr, Y_SIZE: tl.constexpr,
    BX: tl.constexpr, BY: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    xs = pid_x * BX + tl.arange(0, BX)
    ys = pid_y * BY + tl.arange(0, BY)

    x2d = xs[None, :]
    y2d = ys[:, None]

    PI = 3.14159265359
    lx_rot = 30.0
    ly_rot = 0.0
    lx_expan = 0.5
    ly_expan = 0.5
    # Precomputed affine (identical to kernel.h)
    a00 = lx_expan * tl.cos(lx_rot * PI / 180.0)
    a01 = ly_expan * tl.sin(ly_rot * PI / 180.0)
    a10 = lx_expan * tl.sin(lx_rot * PI / 180.0)
    a11 = ly_expan * tl.cos(ly_rot * PI / 180.0)
    det = a00 * a11 - a01 * a10
    # Since det is nonzero for these constants:
    ia00 = a11 / det
    ia01 = -a01 / det
    ia10 = -a10 / det
    ia11 = a00 / det
    ib0 = 0.0
    ib1 = 0.0

    xf = x2d.to(tl.float32)
    yf = y2d.to(tl.float32)
    x_new = ib0 + ia00 * (xf - X_SIZE / 2.0) + ia01 * (yf - Y_SIZE / 2.0) + X_SIZE / 2.0
    y_new = ib1 + ia10 * (xf - X_SIZE / 2.0) + ia11 * (yf - Y_SIZE / 2.0) + Y_SIZE / 2.0

    m = tl.floor(x_new).to(tl.int32)
    n = tl.floor(y_new).to(tl.int32)
    x_frac = x_new - m.to(tl.float32)
    y_frac = y_new - n.to(tl.float32)

    in_bounds = (m >= 0) & (m + 1 < X_SIZE) & (n >= 0) & (n + 1 < Y_SIZE)
    on_edge = (((m + 1 == X_SIZE) & (n >= 0) & (n < Y_SIZE)) |
               ((n + 1 == Y_SIZE) & (m >= 0) & (m < X_SIZE)))

    # Clamp indices for safe gather; result masked later.
    m_safe = tl.where(in_bounds | on_edge, m, 0)
    n_safe = tl.where(in_bounds | on_edge, n, 0)
    m_clamp = tl.minimum(tl.maximum(m_safe, 0), X_SIZE - 1)
    n_clamp = tl.minimum(tl.maximum(n_safe, 0), Y_SIZE - 1)
    m_p1 = tl.minimum(m_clamp + 1, X_SIZE - 1)
    n_p1 = tl.minimum(n_clamp + 1, Y_SIZE - 1)

    p00 = tl.load(src_ptr + n_clamp * X_SIZE + m_clamp).to(tl.float32)
    p01 = tl.load(src_ptr + n_clamp * X_SIZE + m_p1).to(tl.float32)
    p10 = tl.load(src_ptr + n_p1 * X_SIZE + m_clamp).to(tl.float32)
    p11 = tl.load(src_ptr + n_p1 * X_SIZE + m_p1).to(tl.float32)

    gray = (1.0 - y_frac) * ((1.0 - x_frac) * p00 + x_frac * p01) + \
           y_frac * ((1.0 - x_frac) * p10 + x_frac * p11)
    gray_u = gray.to(tl.int32).to(tl.uint16)

    edge_val = tl.load(src_ptr + n_clamp * X_SIZE + m_clamp)
    white = tl.full(x2d.shape, 1, tl.uint16)

    out = tl.where(in_bounds, gray_u, tl.where(on_edge, edge_val, white))

    dst_off = y2d * X_SIZE + x2d
    valid = (x2d < X_SIZE) & (y2d < Y_SIZE)
    tl.store(dst_ptr + dst_off, out, mask=valid)


def affine_reference_np(src):
    """CPU reference — mirror affine-cuda/reference.h."""
    dst = np.zeros_like(src)
    PI = 3.14159265359
    lx_rot = 30.0; ly_rot = 0.0
    lx_expan = 0.5; ly_expan = 0.5
    a00 = lx_expan * math.cos(lx_rot * PI / 180.0)
    a01 = ly_expan * math.sin(ly_rot * PI / 180.0)
    a10 = lx_expan * math.sin(lx_rot * PI / 180.0)
    a11 = ly_expan * math.cos(ly_rot * PI / 180.0)
    det = a00 * a11 - a01 * a10
    ia00 =  a11 / det
    ia01 = -a01 / det
    ia10 = -a10 / det
    ia11 =  a00 / det
    ib0 = ib1 = 0.0

    ys = np.arange(Y_SIZE, dtype=np.float32)
    xs = np.arange(X_SIZE, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)  # (Y, X)
    x_new = ib0 + ia00 * (xg - X_SIZE/2.0) + ia01 * (yg - Y_SIZE/2.0) + X_SIZE/2.0
    y_new = ib1 + ia10 * (xg - X_SIZE/2.0) + ia11 * (yg - Y_SIZE/2.0) + Y_SIZE/2.0
    m = np.floor(x_new).astype(np.int32)
    n = np.floor(y_new).astype(np.int32)
    xf = x_new - m
    yf = y_new - n

    in_b = (m >= 0) & (m + 1 < X_SIZE) & (n >= 0) & (n + 1 < Y_SIZE)
    on_e = (((m + 1 == X_SIZE) & (n >= 0) & (n < Y_SIZE)) |
            ((n + 1 == Y_SIZE) & (m >= 0) & (m < X_SIZE)))

    m_cl = np.clip(m, 0, X_SIZE - 1)
    n_cl = np.clip(n, 0, Y_SIZE - 1)
    m_p1 = np.clip(m + 1, 0, X_SIZE - 1)
    n_p1 = np.clip(n + 1, 0, Y_SIZE - 1)

    p00 = src[n_cl, m_cl].astype(np.float32)
    p01 = src[n_cl, m_p1].astype(np.float32)
    p10 = src[n_p1, m_cl].astype(np.float32)
    p11 = src[n_p1, m_p1].astype(np.float32)
    gray = (1 - yf) * ((1 - xf) * p00 + xf * p01) + yf * ((1 - xf) * p10 + xf * p11)
    gray_u = gray.astype(np.uint16)

    edge_val = src[n_cl, m_cl]
    dst = np.where(in_b, gray_u, np.where(on_e, edge_val, np.uint16(WHITE)))
    return dst.astype(np.uint16)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input image> <output image> <iterations>")
        return 1

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    iters = int(sys.argv[3])

    print("Reading input image...")
    print()
    print("   Reading RAW Image")
    with open(in_path, "rb") as f:
        raw = f.read()
    print(f"   Bytes read = {len(raw)}")
    print()

    src_np = np.frombuffer(raw, dtype=np.uint16).reshape(Y_SIZE, X_SIZE).copy()
    src = torch.from_numpy(src_np).cuda()
    dst = torch.zeros_like(src)

    BX, BY = 16, 16
    grid = (X_SIZE // BX, Y_SIZE // BY)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        affine_kernel[grid](src, dst, X_SIZE, Y_SIZE, BX=BX, BY=BY)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"   Average kernel execution time {elapsed / iters:.9f} (s)")

    dst_np = dst.cpu().numpy()
    ref = affine_reference_np(src_np)
    max_err = int(np.max(np.abs(dst_np.astype(np.int32) - ref.astype(np.int32))))
    print(f"   Max output error is {max_err}")
    print()

    print("   Writing RAW Image")
    with open(out_path, "wb") as f:
        f.write(dst_np.tobytes())
    print(f"   Bytes written = {dst_np.nbytes}")
    print()

    print("PASS" if max_err <= 1 else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
