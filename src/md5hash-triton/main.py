#!/usr/bin/env python3
"""Triton port of the `md5hash` HeCBench benchmark.

Simplification: the CUDA harness sweeps four (byteLength, valsPerByte) sizes
and runs `passes` per size, i.e. up to 4x4=16 keyspace scans, some over 244M
hashes. This port fixes size=1 (byteLength=7, valsPerByte=10 → 10 M keys)
which exercises the full MD5 kernel and result-check contract in far less
time. The Triton kernel is a full 64-round MD5 implementation matched against
Python `hashlib.md5`, so PASS is a proper crypto-correctness check.

Usage: main.py <offload> <passes>
"""
import sys, time, ctypes, hashlib
import torch
import triton
import triton.language as tl


BLOCK = 256

# (constant, shift) for each of the 64 rounds.
CONSTS = [
    (0xd76aa478,  7), (0xe8c7b756, 12), (0x242070db, 17), (0xc1bdceee, 22),
    (0xf57c0faf,  7), (0x4787c62a, 12), (0xa8304613, 17), (0xfd469501, 22),
    (0x698098d8,  7), (0x8b44f7af, 12), (0xffff5bb1, 17), (0x895cd7be, 22),
    (0x6b901122,  7), (0xfd987193, 12), (0xa679438e, 17), (0x49b40821, 22),
    (0xf61e2562,  5), (0xc040b340,  9), (0x265e5a51, 14), (0xe9b6c7aa, 20),
    (0xd62f105d,  5), (0x02441453,  9), (0xd8a1e681, 14), (0xe7d3fbc8, 20),
    (0x21e1cde6,  5), (0xc33707d6,  9), (0xf4d50d87, 14), (0x455a14ed, 20),
    (0xa9e3e905,  5), (0xfcefa3f8,  9), (0x676f02d9, 14), (0x8d2a4c8a, 20),
    (0xfffa3942,  4), (0x8771f681, 11), (0x6d9d6122, 16), (0xfde5380c, 23),
    (0xa4beea44,  4), (0x4bdecfa9, 11), (0xf6bb4b60, 16), (0xbebfbc70, 23),
    (0x289b7ec6,  4), (0xeaa127fa, 11), (0xd4ef3085, 16), (0x04881d05, 23),
    (0xd9d4d039,  4), (0xe6db99e5, 11), (0x1fa27cf8, 16), (0xc4ac5665, 23),
    (0xf4292244,  6), (0x432aff97, 10), (0xab9423a7, 15), (0xfc93a039, 21),
    (0x655b59c3,  6), (0x8f0ccc92, 10), (0xffeff47d, 15), (0x85845dd1, 21),
    (0x6fa87e4f,  6), (0xfe2ce6e0, 10), (0xa3014314, 15), (0x4e0811a1, 21),
    (0xf7537e82,  6), (0xbd3af235, 10), (0x2ad7d2bb, 15), (0xeb86d391, 21),
]
# W-selector: 0 -> const 0, 1 -> W0, 2 -> W1, 3 -> WL
W_SEL = [
    1,2,0,0, 0,0,0,0, 0,0,0,0, 0,0,3,0,
    2,0,0,1, 0,0,0,0, 0,3,0,0, 0,0,0,0,
    0,0,0,3, 2,0,0,0, 0,1,0,0, 0,0,0,0,
    1,0,3,0, 0,0,0,2, 0,0,0,0, 0,0,0,0,
]
FUNC_SEL = [0]*16 + [1]*16 + [2]*16 + [3]*16


def _emit_kernel_source():
    """Build a per-round-unrolled MD5 kernel source string.

    Triton's compile-time unrolling is limited: we can't switch on a per-round
    constexpr inside a helper. So we generate the kernel code textually with
    each round's constants baked in. The runtime is a big single kernel that
    the JIT still handles fine because everything is straight-line arithmetic.
    """
    lines = []
    lines.append(
        "@triton.jit\n"
        "def md5_kernel(\n"
        "    found_idx_ptr, found_key_ptr, found_dig_ptr,\n"
        "    keyspace, valsPerByte,\n"
        "    sd0, sd1, sd2, sd3,\n"
        "    BLOCK: tl.constexpr, VPB: tl.constexpr, BLEN: tl.constexpr,\n"
        "):\n"
        "    pid = tl.program_id(0)\n"
        "    tid = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    start_idx = tid * valsPerByte\n"
        "    idx = start_idx\n"
        "    v0 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v1 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v2 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v3 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v4 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v5 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v6 = idx % valsPerByte; idx = idx // valsPerByte\n"
        "    v7 = idx % valsPerByte\n"
        "    v0 = v0.to(tl.uint32); v1 = v1.to(tl.uint32); v2 = v2.to(tl.uint32); v3 = v3.to(tl.uint32)\n"
        "    v4 = v4.to(tl.uint32); v5 = v5.to(tl.uint32); v6 = v6.to(tl.uint32); v7 = v7.to(tl.uint32)\n"
        "    for j in tl.static_range(VPB):\n"
        "        active = (start_idx + j) < keyspace\n"
        "        W0 = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)\n"
        "        W1 = v4 | (v5 << 8) | (v6 << 16) | (v7 << 24)\n"
        # Padding: byteLength=7 places 0x80 at byte 7 of W1
        "        W1 = W1 | 0x80000000\n"
        "        WL = tl.full(W0.shape, BLEN * 8, tl.uint32)\n"
        "        zero = tl.full(W0.shape, 0, tl.uint32)\n"
        "        a = tl.full(W0.shape, 0x67452301, tl.uint32)\n"
        "        b = tl.full(W0.shape, 0xefcdab89, tl.uint32)\n"
        "        c = tl.full(W0.shape, 0x98badcfe, tl.uint32)\n"
        "        d = tl.full(W0.shape, 0x10325476, tl.uint32)\n"
    )
    def w_of(k):
        return {0: "zero", 1: "W0", 2: "W1", 3: "WL"}[W_SEL[k]]
    # Round ordering: v/x/y/z rotates a,b,c,d in the pattern a/b/c/d, d/a/b/c, c/d/a/b, b/c/d/a
    shuffles = [("a","b","c","d"), ("d","a","b","c"), ("c","d","a","b"), ("b","c","d","a")]
    for k in range(64):
        v, x, y, z = shuffles[k % 4]
        K, R = CONSTS[k]
        w = w_of(k)
        if FUNC_SEL[k] == 0:
            fn = f"(({x} & {y}) | ((~{x}) & {z}))"
        elif FUNC_SEL[k] == 1:
            fn = f"(({x} & {z}) | ((~{z}) & {y}))"
        elif FUNC_SEL[k] == 2:
            fn = f"({x} ^ {y} ^ {z})"
        else:
            fn = f"({y} ^ ({x} | (~{z})))"
        lines.append(
            f"        {v} = {v} + {fn} + {w} + tl.full(W0.shape, 0x{K:08x}, tl.uint32)\n"
            f"        {v} = ({v} << {R}) | ({v} >> {32 - R})\n"
            f"        {v} = {x} + {v}\n"
        )
    lines.append(
        "        h0 = a + tl.full(a.shape, 0x67452301, tl.uint32)\n"
        "        h1 = b + tl.full(a.shape, 0xefcdab89, tl.uint32)\n"
        "        h2 = c + tl.full(a.shape, 0x98badcfe, tl.uint32)\n"
        "        h3 = d + tl.full(a.shape, 0x10325476, tl.uint32)\n"
        "        match = (h0 == sd0) & (h1 == sd1) & (h2 == sd2) & (h3 == sd3) & active\n"
        "        idx_val = (start_idx + j).to(tl.int32)\n"
        "        big = tl.full(idx_val.shape, 0x7fffffff, tl.int32)\n"
        "        idx_broadcast = tl.where(match, idx_val, big)\n"
        "        min_idx = tl.min(idx_broadcast, axis=0)\n"
        "        tl.atomic_min(found_idx_ptr, min_idx)\n"
        "        v0 = ((v0 + tl.full(v0.shape, 1, tl.uint32)) % valsPerByte.to(tl.uint32))\n"
    )
    return "".join(lines)


# Build the kernel source once at import time, write it to a sibling file,
# then import so Triton can inspect its source.
import os, importlib.util
_KERNEL_SRC = _emit_kernel_source()
_KERNEL_PATH = os.path.join(os.path.dirname(__file__), "_md5_kernel_gen.py")
_HEADER = (
    "import triton\n"
    "import triton.language as tl\n\n"
)
with open(_KERNEL_PATH, "w") as _f:
    _f.write(_HEADER + _KERNEL_SRC)
_spec = importlib.util.spec_from_file_location("_md5_kernel_gen", _KERNEL_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
md5_kernel = _mod.md5_kernel


def md5_hashlib(key: bytes) -> tuple:
    """Return the 4-word MD5 digest (little-endian) of key."""
    h = hashlib.md5(key).digest()
    return tuple(int.from_bytes(h[i*4:(i+1)*4], "little") for i in range(4))


def index_to_key(index, byte_length, vals_per_byte):
    key = [0] * 8
    for i in range(byte_length):
        key[i] = index % vals_per_byte
        index //= vals_per_byte
    return bytes(key)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <offload> <passes>")
        return 1
    offload = int(sys.argv[1])
    passes = int(sys.argv[2])

    # Simplified: fix size=1 (byteLength=7, valsPerByte=10)
    byte_length = 7
    vals_per_byte = 10
    keyspace = vals_per_byte ** byte_length  # 10^7 = 10M
    print(f"Searching keys of length {byte_length} bytes and {vals_per_byte} values per byte")
    print(f"|keyspace| = {keyspace} ({keyspace//1_000_000}M)")

    # Match srandom(12345) sequence approximately; use libc's random().
    libc = ctypes.CDLL("libc.so.6")
    libc.srandom(ctypes.c_uint(12345))
    libc.random.restype = ctypes.c_long

    for p in range(passes):
        rand_idx = libc.random() % keyspace
        rand_key = index_to_key(rand_idx, byte_length, vals_per_byte)
        rand_digest = md5_hashlib(rand_key[:byte_length])
        print(f"\n--- iteration {p} ---")
        print("Looking for random key:")
        print(f" randomIndex = {rand_idx}")
        print(f" randomKey   = 0x{rand_key.hex().upper()}")
        print(f" randomDigest= " + "".join(
            f"{b:02X}" for w in rand_digest for b in w.to_bytes(4, 'little')
        ))

        d_found_idx = torch.tensor([0x7fffffff], dtype=torch.int32, device="cuda")
        d_found_key = torch.zeros(8, dtype=torch.uint8, device="cuda")
        d_found_dig = torch.zeros(4, dtype=torch.uint32, device="cuda")

        nthreads_per_block = BLOCK
        total_threads = (keyspace + vals_per_byte - 1) // vals_per_byte
        nblocks = (total_threads + nthreads_per_block - 1) // nthreads_per_block

        t0 = time.perf_counter()
        md5_kernel[(nblocks,)](
            d_found_idx, d_found_key, d_found_dig,
            keyspace, vals_per_byte,
            rand_digest[0], rand_digest[1], rand_digest[2], rand_digest[3],
            BLOCK=BLOCK, VPB=vals_per_byte, BLEN=byte_length,
        )
        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000
        found_idx = int(d_found_idx.cpu().item())
        rate = keyspace / (t_ms / 1000) / 1e9 if t_ms > 0 else 0
        print(f"time = {t_ms:.0f} ms, rate = {rate:.4f} GHash/sec")

        # Verify: reconstruct found_key/digest from found_idx and check
        if found_idx == 0x7fffffff or found_idx < 0:
            print("\nERROR: could not find a match.")
            print("FAIL")
            continue
        found_key = index_to_key(found_idx, byte_length, vals_per_byte)
        found_dig = md5_hashlib(found_key[:byte_length])

        print(f"\nSuccessfully found match (index, key, hash):")
        print(f" foundIndex  = {found_idx}")
        print(f" foundKey    = 0x{found_key.hex().upper()}")
        print(f" foundDigest = " + "".join(
            f"{b:02X}" for w in found_dig for b in w.to_bytes(4, 'little')
        ))
        ok = (found_idx == rand_idx and found_dig == rand_digest)
        print("PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
