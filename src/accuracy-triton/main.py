#!/usr/bin/env python3
"""Triton port of the `accuracy` HeCBench benchmark.

Computes top-K classification accuracy. Matches the semantics of accuracy-cuda:
count = number of rows where the ground-truth label's prediction rank is <= top_k.

Usage: main.py <nrows> <ndims> <top_k> <repeat>
"""
import sys, time
import torch
import triton
import triton.language as tl

@triton.jit
def top_k_count_kernel(
    data_ptr, label_ptr, count_ptr,
    nrows, ndims, top_k,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= nrows:
        return
    label = tl.load(label_ptr + row)
    label_pred = tl.load(data_ptr + row * stride_row + label)

    ngt = 0
    for start in range(0, ndims, BLOCK):
        col = start + tl.arange(0, BLOCK)
        mask = col < ndims
        pred = tl.load(data_ptr + row * stride_row + col, mask=mask, other=0.0)
        gt = (pred > label_pred) | ((pred == label_pred) & (col <= label))
        ngt += tl.sum(tl.where(mask & gt, 1, 0))

    if ngt <= top_k:
        tl.atomic_add(count_ptr, 1)


def reference_count(data: torch.Tensor, label: torch.Tensor, top_k: int) -> int:
    # CPU/torch reference — matches accuracy-cuda/reference.h
    nrows, ndims = data.shape
    col_idx = torch.arange(ndims, device=data.device).unsqueeze(0).expand(nrows, ndims)
    label_pred = data.gather(1, label.unsqueeze(1))
    gt = (data > label_pred) | ((data == label_pred) & (col_idx <= label.unsqueeze(1)))
    ngt = gt.sum(dim=1)
    return int((ngt <= top_k).sum().item())


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <number of rows> <number of columns> <top K> <repeat>")
        return 1
    nrows, ndims, top_k, repeat = map(int, sys.argv[1:5])

    # Match accuracy-cuda seeding: srand(123) + rand() % ndims for labels,
    # std::default_random_engine(123) + uniform<float>(0,1) for data.
    # These will differ numerically from the C++ generator, but the top-K
    # verifier is order-insensitive so the count still agrees with the on-GPU
    # kernel (both see the same data).
    gen = torch.Generator(device="cuda").manual_seed(123)
    label = torch.randint(0, ndims, (nrows,), device="cuda", generator=gen, dtype=torch.int32)
    data  = torch.rand((nrows, ndims), device="cuda", generator=gen, dtype=torch.float32)

    count_ref = reference_count(data, label.long(), top_k)

    BLOCK = 1024
    count = torch.zeros(1, device="cuda", dtype=torch.int32)

    for ngrid in (nrows // 4, nrows // 2, 3 * nrows // 4, nrows):
        print(f"Grid size is {ngrid}")

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            count.zero_()
            top_k_count_kernel[(nrows,)](
                data, label, count,
                nrows, ndims, top_k,
                data.stride(0),
                BLOCK=BLOCK,
            )
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - start) * 1e6 / repeat

        print(f"Average execution time of accuracy kernel: {elapsed_us:.6f} (us)")
        ok = int(count.item()) == count_ref
        print("PASS" if ok else "FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
