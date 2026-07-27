#!/usr/bin/env python3
"""Triton port of the `sobel` HeCBench benchmark.

3x3 Sobel edge-detection on an RGBA uchar4 image. Because the reference BMP
input (SobelFilter_Input.bmp) is not shipped with HeCBench, this port
synthesizes a deterministic 512x512 RGBA image when the file is missing and
runs the kernel on that. Compares against a torch-CPU reference using the
same relative-L2 tolerance the CUDA harness uses.

Usage: main.py <path to bmp | ignored if missing> <iterations>
"""
import sys, time, os, math
import torch
import triton
import triton.language as tl


BLOCK_X = 16
BLOCK_Y = 16


@triton.jit
def sobel_channel(
    inp_ptr, out_ptr, width, height, channel,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]
    y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]

    inner = (x >= 1) & (x < width - 1) & (y >= 1) & (y < height - 1)

    base = (y * width + x) * 4 + channel
    stride = 4
    row = 4 * width
    i00 = tl.load(inp_ptr + base - row - stride, mask=inner, other=0).to(tl.float32)
    i01 = tl.load(inp_ptr + base - row,          mask=inner, other=0).to(tl.float32)
    i02 = tl.load(inp_ptr + base - row + stride, mask=inner, other=0).to(tl.float32)
    i10 = tl.load(inp_ptr + base - stride,       mask=inner, other=0).to(tl.float32)
    i12 = tl.load(inp_ptr + base + stride,       mask=inner, other=0).to(tl.float32)
    i20 = tl.load(inp_ptr + base + row - stride, mask=inner, other=0).to(tl.float32)
    i21 = tl.load(inp_ptr + base + row,          mask=inner, other=0).to(tl.float32)
    i22 = tl.load(inp_ptr + base + row + stride, mask=inner, other=0).to(tl.float32)

    Gx = i00 + 2.0 * i10 + i20 - i02 - 2.0 * i12 - i22
    Gy = i00 + 2.0 * i01 + i02 - i20 - 2.0 * i21 - i22
    m = tl.sqrt(Gx * Gx + Gy * Gy) * 0.5
    m = tl.where(m > 255.0, 255.0, m)
    m = tl.where(m < 0.0, 0.0, m)
    m_u8 = m.to(tl.uint8)

    tl.store(out_ptr + base, m_u8, mask=inner)


def load_bmp_or_synth(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        # Very small BMP parser for 24- or 32-bit uncompressed BMPs
        assert data[:2] == b"BM"
        pixel_offset = int.from_bytes(data[10:14], "little")
        width = int.from_bytes(data[18:22], "little")
        height_raw = int.from_bytes(data[22:26], "little", signed=True)
        bpp = int.from_bytes(data[28:30], "little")
        height = abs(height_raw)
        row_bytes = ((bpp * width + 31) // 32) * 4
        img = torch.zeros((height, width, 4), dtype=torch.uint8)
        for row in range(height):
            r = height - 1 - row if height_raw > 0 else row
            base = pixel_offset + row * row_bytes
            for col in range(width):
                if bpp == 32:
                    b = data[base + col*4 + 0]
                    g = data[base + col*4 + 1]
                    rr = data[base + col*4 + 2]
                    aa = data[base + col*4 + 3]
                else:
                    b = data[base + col*3 + 0]
                    g = data[base + col*3 + 1]
                    rr = data[base + col*3 + 2]
                    aa = 255
                img[r, col, 0] = rr; img[r, col, 1] = g
                img[r, col, 2] = b;  img[r, col, 3] = aa
        return img
    # Synthetic 512x512 image, deterministic.
    print(f"[note] {path} not found — using synthetic 512x512 RGBA input")
    torch.manual_seed(0)
    width, height = 512, 512
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    r = ((xx * 5 + yy * 3) & 0xFF).to(torch.uint8)
    g = ((xx.pow(2) + yy) & 0xFF).to(torch.uint8)
    b = ((xx ^ yy) & 0xFF).to(torch.uint8)
    a = torch.full_like(r, 255)
    return torch.stack([r, g, b, a], dim=-1)


def reference_torch(img):
    """Apply Sobel filter on CPU using torch ops (matches reference.cu)."""
    h, w, _ = img.shape
    f = img.to(torch.float32)  # (h, w, 4)
    out = torch.zeros_like(img)
    # slice interior
    r_c = f[1:-1, 1:-1]
    r_l = f[1:-1, :-2]; r_r = f[1:-1, 2:]
    r_u = f[:-2, 1:-1]; r_d = f[2:, 1:-1]
    r_ul = f[:-2, :-2]; r_ur = f[:-2, 2:]
    r_dl = f[2:, :-2];  r_dr = f[2:, 2:]
    Gx = r_ul + 2*r_l + r_dl - r_ur - 2*r_r - r_dr
    Gy = r_ul + 2*r_u + r_ur - r_dl - 2*r_d - r_dr
    mag = (Gx * Gx + Gy * Gy).sqrt() * 0.5
    mag = mag.clamp(0, 255).to(torch.uint8)
    out[1:-1, 1:-1] = mag
    return out


def compare(ref, dev):
    ref_f = ref.to(torch.float32).flatten()
    dev_f = dev.to(torch.float32).flatten()
    diff = ref_f[1:] - dev_f[1:]
    error = float((diff * diff).sum().sqrt())
    ref_n = float((ref_f[1:] * ref_f[1:]).sum().sqrt())
    if ref_n < 1e-7:
        return False
    return error / ref_n < 1e-6


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path to bmp> <repeat>")
        return 1
    path = sys.argv[1]
    iters = int(sys.argv[2])

    img = load_bmp_or_synth(path)  # (h, w, 4) uint8
    h, w, _ = img.shape
    print(f"Image height = {h} and width = {w}")

    d_in = img.cuda().contiguous().view(-1)
    d_out = torch.zeros_like(d_in)

    grid = (w // BLOCK_X, h // BLOCK_Y)
    print(f"Executing kernel for {iters} iterations")
    print("-------------------------------------------")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for ch in range(4):
            sobel_channel[grid](d_in, d_out, w, h, ch, BLOCK_X=BLOCK_X, BLOCK_Y=BLOCK_Y)
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - t0) * 1e6 / iters
    print(f"Average kernel execution time: {dt_us:f} (us)")

    dev_img = d_out.cpu().view(h, w, 4)
    ref_img = reference_torch(img)
    ok = compare(ref_img, dev_img)
    print("PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
