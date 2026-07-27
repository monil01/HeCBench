#!/usr/bin/env python3
"""Triton port of the `softmax` HeCBench benchmark.

Row-wise softmax over ``numSlice`` rows of ``sliceSize`` elements each.
kernel==0 -> row-per-program-id
kernel==1 -> warp-sized program (same numerics; Triton runs one program per row)
"""
import sys, time
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(src_ptr, dst_ptr, numSlice, sliceSize, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= numSlice:
        return
    row_start = row * sliceSize
    offs = tl.arange(0, BLOCK)

    # First pass: max
    row_max = -float("inf")
    for start in range(0, sliceSize, BLOCK):
        idx = start + offs
        m = idx < sliceSize
        v = tl.load(src_ptr + row_start + idx, mask=m, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(v, axis=0))

    # Second pass: sum of exp
    row_sum = 0.0
    for start in range(0, sliceSize, BLOCK):
        idx = start + offs
        m = idx < sliceSize
        v = tl.load(src_ptr + row_start + idx, mask=m, other=0.0)
        e = tl.exp(v - row_max)
        row_sum += tl.sum(tl.where(m, e, 0.0), axis=0)

    # Third pass: write normalized
    for start in range(0, sliceSize, BLOCK):
        idx = start + offs
        m = idx < sliceSize
        v = tl.load(src_ptr + row_start + idx, mask=m, other=0.0)
        e = tl.exp(v - row_max) / row_sum
        tl.store(dst_ptr + row_start + idx, e, mask=m)


def softmax_ref(src: torch.Tensor) -> torch.Tensor:
    # naive row-wise softmax on CPU
    m = src.max(dim=1, keepdim=True).values
    e = (src - m).exp()
    return e / e.sum(dim=1, keepdim=True)


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <numSlice> <sliceSize> <impl 0|1> <repeat>")
        return 1
    numSlice = int(sys.argv[1])
    sliceSize = int(sys.argv[2])
    kernel = int(sys.argv[3])
    repeat = int(sys.argv[4])

    torch.manual_seed(2)
    inp = torch.randint(0, 13, (numSlice, sliceSize), dtype=torch.int32).float()
    d_in = inp.cuda().contiguous()
    d_out = torch.empty_like(d_in)

    BLOCK = 1024 if sliceSize > 1024 else max(1, triton.next_power_of_2(sliceSize))

    # warmup
    softmax_kernel[(numSlice,)](d_in, d_out, numSlice, sliceSize, BLOCK=BLOCK)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        softmax_kernel[(numSlice,)](d_in, d_out, numSlice, sliceSize, BLOCK=BLOCK)
    torch.cuda.synchronize()
    kt_ms = (time.perf_counter() - t0) * 1000.0 / repeat
    print(f"Average kernel execution time: {kt_ms:f} (ms)")

    got = d_out.cpu()
    ref = softmax_ref(inp)
    err = float((got - ref).abs().max().item())
    ok = err < 1e-3
    if not ok:
        idx = (got - ref).abs().argmax()
        print(f"@index {idx.item()} host: {ref.view(-1)[idx].item()} device: {got.view(-1)[idx].item()}")
    print("PASS" if ok else "FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
