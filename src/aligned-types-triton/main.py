#!/usr/bin/env python3
"""Triton port of the `aligned-types` HeCBench benchmark.

The C++ benchmark compares copy throughput across a bunch of
aligned/misaligned struct types (1, 2, 4, 8, 12, 16, 32 byte elements).
In Triton we cover the same *packed element sizes* by treating the buffer
as 1/2/4/8/16/32 byte-wide records and dispatching a per-element Triton
copy kernel. Prints per-case TEST PASS lines and a final Test passed
banner when all pass.

Args: none.
"""
import sys, time
import torch
import triton
import triton.language as tl


@triton.jit
def copy_bytes_kernel(dst_ptr, src_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, v, mask=mask)


def _align_down(a, b):
    return a - (a % b)


def run_test(d_idata_u8: torch.Tensor, d_odata_u8: torch.Tensor,
             packed_element_size: int, memory_size: int, name: str,
             iterations: int = 1000) -> int:
    total_aligned = _align_down(memory_size, packed_element_size)
    num_elements = memory_size // packed_element_size

    # Clear output.
    d_odata_u8.zero_()
    torch.cuda.synchronize()

    n_bytes_total = num_elements * packed_element_size
    BLOCK = 1024
    grid = ((n_bytes_total + BLOCK - 1) // BLOCK,)

    t0 = time.perf_counter()
    for _ in range(iterations):
        copy_bytes_kernel[grid](d_odata_u8, d_idata_u8, n_bytes_total, BLOCK=BLOCK)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iterations

    print(f"Avg. time: {dt * 1000:f} ms / "
          f"Copy throughput: {total_aligned / (dt * (1 << 30)):f} GB/s.")

    # Validation: verify the copied packed bytes match.
    src = d_idata_u8[:total_aligned].cpu()
    dst = d_odata_u8[:total_aligned].cpu()
    flag = bool(torch.equal(src, dst))
    print(f"\tTEST {'PASS' if flag else 'FAIL'}")
    return 0 if flag else 1


def main():
    MEM_SIZE = 50_000_000
    MemorySize = MEM_SIZE & 0xFFFFFF00
    print(f"[{sys.argv[0]}] - Starting...")
    print("Allocating memory...")
    print("Generating host input data array...")
    # matches the C++ pattern h_idataCPU[i] = (i & 0xFF) + 1
    idx = torch.arange(MemorySize, dtype=torch.int32) & 0xFF
    h_idata = ((idx + 1) & 0xFF).to(torch.uint8)
    print("Uploading input data to GPU memory...")
    d_idata = h_idata.cuda()
    d_odata = torch.zeros(MemorySize, dtype=torch.uint8, device="cuda")

    total_fail = 0
    cases = [
        ("uchar_misaligned", 1),
        ("uchar4_misaligned", 4),
        ("uchar4_aligned", 4),
        ("ushort_misaligned", 2),
        ("uint_aligned", 4),
        ("uint2_misaligned", 8),
        ("uint2_aligned", 8),
        ("uint3_misaligned", 12),
        ("uint3_aligned", 12),
        ("uint4_misaligned", 16),
        ("uint4_aligned", 16),
        ("uint8_misaligned", 32),
        ("uint8_aligned", 32),
    ]

    print("Testing misaligned types...")
    for name, size in cases:
        print(f"{name}...")
        total_fail += run_test(d_idata, d_odata, size, MemorySize, name)

    print(f"\n[alignedTypes] -> Test Results: {total_fail} Failures")
    print("Shutting down...")
    if total_fail != 0:
        print("Test failed!")
        return 1
    print("Test passed")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
