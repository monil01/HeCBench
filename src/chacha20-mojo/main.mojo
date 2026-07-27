# Mojo GPU port of the `chacha20` HeCBench benchmark (simplified).
#
# ChaCha20 keystream generator. Each GPU thread produces one 64-byte
# block (state[0..15] as uint32) using its own counter. The quarter-
# round + 20-round core is copied from chacha20-cuda/chacha20.h.
# Compared against a host reference implementation.
#
# Usage: main.mojo <repeat>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


comptime NBLOCKS: Int = 4096  # number of 64-byte blocks


def rotl32(x: UInt32, n: UInt32) -> UInt32:
    return (x << n) | (x >> (UInt32(32) - n))


def qr_gpu(state: UnsafePointer[UInt32, MutAnyOrigin],
           a: Int, b: Int, c: Int, d: Int):
    state[a] = state[a] + state[b]; state[d] = rotl32(state[d] ^ state[a], UInt32(16))
    state[c] = state[c] + state[d]; state[b] = rotl32(state[b] ^ state[c], UInt32(12))
    state[a] = state[a] + state[b]; state[d] = rotl32(state[d] ^ state[a], UInt32(8))
    state[c] = state[c] + state[d]; state[b] = rotl32(state[b] ^ state[c], UInt32(7))


def chacha_kernel(
        keystream: UnsafePointer[UInt32, MutAnyOrigin],
        key0: UInt32, key1: UInt32, key2: UInt32, key3: UInt32,
        key4: UInt32, key5: UInt32, key6: UInt32, key7: UInt32,
        nonce0: UInt32, nonce1: UInt32,
        nblocks: Int32):
    var bid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if bid >= Int(nblocks): return

    # ChaCha20 constants "expand 32-byte k"
    var c0: UInt32 = UInt32(0x61707865)
    var c1: UInt32 = UInt32(0x3320646e)
    var c2: UInt32 = UInt32(0x79622d32)
    var c3: UInt32 = UInt32(0x6b206574)

    # Local state[16]
    var s = InlineArray[UInt32, 16](fill=UInt32(0))
    s[0] = c0; s[1] = c1; s[2] = c2; s[3] = c3
    s[4] = key0; s[5] = key1; s[6] = key2; s[7] = key3
    s[8] = key4; s[9] = key5; s[10] = key6; s[11] = key7
    s[12] = UInt32(bid); s[13] = UInt32(0)   # counter (bid < 2^32)
    s[14] = nonce0; s[15] = nonce1

    var r = InlineArray[UInt32, 16](fill=UInt32(0))
    for i in range(16): r[i] = s[i]

    # 10 double-rounds = 20 rounds
    for _ in range(10):
        # Column rounds
        r[0] = r[0] + r[4]; r[12] = rotl32(r[12] ^ r[0], UInt32(16))
        r[8] = r[8] + r[12]; r[4]  = rotl32(r[4]  ^ r[8], UInt32(12))
        r[0] = r[0] + r[4]; r[12] = rotl32(r[12] ^ r[0], UInt32(8))
        r[8] = r[8] + r[12]; r[4]  = rotl32(r[4]  ^ r[8], UInt32(7))

        r[1] = r[1] + r[5]; r[13] = rotl32(r[13] ^ r[1], UInt32(16))
        r[9] = r[9] + r[13]; r[5]  = rotl32(r[5]  ^ r[9], UInt32(12))
        r[1] = r[1] + r[5]; r[13] = rotl32(r[13] ^ r[1], UInt32(8))
        r[9] = r[9] + r[13]; r[5]  = rotl32(r[5]  ^ r[9], UInt32(7))

        r[2] = r[2] + r[6]; r[14] = rotl32(r[14] ^ r[2], UInt32(16))
        r[10] = r[10] + r[14]; r[6]  = rotl32(r[6]  ^ r[10], UInt32(12))
        r[2] = r[2] + r[6]; r[14] = rotl32(r[14] ^ r[2], UInt32(8))
        r[10] = r[10] + r[14]; r[6]  = rotl32(r[6]  ^ r[10], UInt32(7))

        r[3] = r[3] + r[7]; r[15] = rotl32(r[15] ^ r[3], UInt32(16))
        r[11] = r[11] + r[15]; r[7]  = rotl32(r[7]  ^ r[11], UInt32(12))
        r[3] = r[3] + r[7]; r[15] = rotl32(r[15] ^ r[3], UInt32(8))
        r[11] = r[11] + r[15]; r[7]  = rotl32(r[7]  ^ r[11], UInt32(7))

        # Diagonal rounds
        r[0] = r[0] + r[5]; r[15] = rotl32(r[15] ^ r[0], UInt32(16))
        r[10] = r[10] + r[15]; r[5]  = rotl32(r[5]  ^ r[10], UInt32(12))
        r[0] = r[0] + r[5]; r[15] = rotl32(r[15] ^ r[0], UInt32(8))
        r[10] = r[10] + r[15]; r[5]  = rotl32(r[5]  ^ r[10], UInt32(7))

        r[1] = r[1] + r[6]; r[12] = rotl32(r[12] ^ r[1], UInt32(16))
        r[11] = r[11] + r[12]; r[6]  = rotl32(r[6]  ^ r[11], UInt32(12))
        r[1] = r[1] + r[6]; r[12] = rotl32(r[12] ^ r[1], UInt32(8))
        r[11] = r[11] + r[12]; r[6]  = rotl32(r[6]  ^ r[11], UInt32(7))

        r[2] = r[2] + r[7]; r[13] = rotl32(r[13] ^ r[2], UInt32(16))
        r[8]  = r[8]  + r[13]; r[7]  = rotl32(r[7]  ^ r[8],  UInt32(12))
        r[2] = r[2] + r[7]; r[13] = rotl32(r[13] ^ r[2], UInt32(8))
        r[8]  = r[8]  + r[13]; r[7]  = rotl32(r[7]  ^ r[8],  UInt32(7))

        r[3] = r[3] + r[4]; r[14] = rotl32(r[14] ^ r[3], UInt32(16))
        r[9]  = r[9]  + r[14]; r[4]  = rotl32(r[4]  ^ r[9],  UInt32(12))
        r[3] = r[3] + r[4]; r[14] = rotl32(r[14] ^ r[3], UInt32(8))
        r[9]  = r[9]  + r[14]; r[4]  = rotl32(r[4]  ^ r[9],  UInt32(7))

    for i in range(16):
        keystream[bid * 16 + i] = r[i] + s[i]


def rotl32_h(x: UInt32, n: UInt32) -> UInt32:
    return (x << n) | (x >> (UInt32(32) - n))


def chacha_host_block(bid: Int,
                      key0: UInt32, key1: UInt32, key2: UInt32, key3: UInt32,
                      key4: UInt32, key5: UInt32, key6: UInt32, key7: UInt32,
                      nonce0: UInt32, nonce1: UInt32,
                      out_p: UnsafePointer[UInt32, MutAnyOrigin]):
    var c0: UInt32 = UInt32(0x61707865)
    var c1: UInt32 = UInt32(0x3320646e)
    var c2: UInt32 = UInt32(0x79622d32)
    var c3: UInt32 = UInt32(0x6b206574)
    var s: List[UInt32] = [c0, c1, c2, c3,
                           key0, key1, key2, key3,
                           key4, key5, key6, key7,
                           UInt32(bid), UInt32(0), nonce0, nonce1]
    var r: List[UInt32] = s.copy()
    for _ in range(10):
        # Column
        r[0] = r[0] + r[4]; r[12] = rotl32_h(r[12] ^ r[0], UInt32(16))
        r[8] = r[8] + r[12]; r[4]  = rotl32_h(r[4]  ^ r[8], UInt32(12))
        r[0] = r[0] + r[4]; r[12] = rotl32_h(r[12] ^ r[0], UInt32(8))
        r[8] = r[8] + r[12]; r[4]  = rotl32_h(r[4]  ^ r[8], UInt32(7))
        r[1] = r[1] + r[5]; r[13] = rotl32_h(r[13] ^ r[1], UInt32(16))
        r[9] = r[9] + r[13]; r[5]  = rotl32_h(r[5]  ^ r[9], UInt32(12))
        r[1] = r[1] + r[5]; r[13] = rotl32_h(r[13] ^ r[1], UInt32(8))
        r[9] = r[9] + r[13]; r[5]  = rotl32_h(r[5]  ^ r[9], UInt32(7))
        r[2] = r[2] + r[6]; r[14] = rotl32_h(r[14] ^ r[2], UInt32(16))
        r[10] = r[10] + r[14]; r[6]  = rotl32_h(r[6]  ^ r[10], UInt32(12))
        r[2] = r[2] + r[6]; r[14] = rotl32_h(r[14] ^ r[2], UInt32(8))
        r[10] = r[10] + r[14]; r[6]  = rotl32_h(r[6]  ^ r[10], UInt32(7))
        r[3] = r[3] + r[7]; r[15] = rotl32_h(r[15] ^ r[3], UInt32(16))
        r[11] = r[11] + r[15]; r[7]  = rotl32_h(r[7]  ^ r[11], UInt32(12))
        r[3] = r[3] + r[7]; r[15] = rotl32_h(r[15] ^ r[3], UInt32(8))
        r[11] = r[11] + r[15]; r[7]  = rotl32_h(r[7]  ^ r[11], UInt32(7))
        # Diagonal
        r[0] = r[0] + r[5]; r[15] = rotl32_h(r[15] ^ r[0], UInt32(16))
        r[10] = r[10] + r[15]; r[5]  = rotl32_h(r[5]  ^ r[10], UInt32(12))
        r[0] = r[0] + r[5]; r[15] = rotl32_h(r[15] ^ r[0], UInt32(8))
        r[10] = r[10] + r[15]; r[5]  = rotl32_h(r[5]  ^ r[10], UInt32(7))
        r[1] = r[1] + r[6]; r[12] = rotl32_h(r[12] ^ r[1], UInt32(16))
        r[11] = r[11] + r[12]; r[6]  = rotl32_h(r[6]  ^ r[11], UInt32(12))
        r[1] = r[1] + r[6]; r[12] = rotl32_h(r[12] ^ r[1], UInt32(8))
        r[11] = r[11] + r[12]; r[6]  = rotl32_h(r[6]  ^ r[11], UInt32(7))
        r[2] = r[2] + r[7]; r[13] = rotl32_h(r[13] ^ r[2], UInt32(16))
        r[8]  = r[8]  + r[13]; r[7]  = rotl32_h(r[7]  ^ r[8],  UInt32(12))
        r[2] = r[2] + r[7]; r[13] = rotl32_h(r[13] ^ r[2], UInt32(8))
        r[8]  = r[8]  + r[13]; r[7]  = rotl32_h(r[7]  ^ r[8],  UInt32(7))
        r[3] = r[3] + r[4]; r[14] = rotl32_h(r[14] ^ r[3], UInt32(16))
        r[9]  = r[9]  + r[14]; r[4]  = rotl32_h(r[4]  ^ r[9],  UInt32(12))
        r[3] = r[3] + r[4]; r[14] = rotl32_h(r[14] ^ r[3], UInt32(8))
        r[9]  = r[9]  + r[14]; r[4]  = rotl32_h(r[4]  ^ r[9],  UInt32(7))
    for i in range(16):
        out_p[bid * 16 + i] = r[i] + s[i]


def main() raises:
    var args = argv()
    var repeat = Int(atol(args[1])) if len(args) > 1 else 10

    # Fixed key/nonce
    var key0: UInt32 = UInt32(0x03020100)
    var key1: UInt32 = UInt32(0x07060504)
    var key2: UInt32 = UInt32(0x0b0a0908)
    var key3: UInt32 = UInt32(0x0f0e0d0c)
    var key4: UInt32 = UInt32(0x13121110)
    var key5: UInt32 = UInt32(0x17161514)
    var key6: UInt32 = UInt32(0x1b1a1918)
    var key7: UInt32 = UInt32(0x1f1e1d1c)
    var nonce0: UInt32 = UInt32(0x03020100)
    var nonce1: UInt32 = UInt32(0x07060504)

    var ctx = DeviceContext()
    var d_out = ctx.enqueue_create_buffer[DType.uint32](NBLOCKS * 16)
    var ref_out = ctx.enqueue_create_buffer[DType.uint32](NBLOCKS * 16)

    comptime BLOCK: Int = 128
    var blocks = (NBLOCKS + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(repeat):
        ctx.enqueue_function[func=chacha_kernel](
            d_out.unsafe_ptr(),
            key0, key1, key2, key3, key4, key5, key6, key7,
            nonce0, nonce1, Int32(NBLOCKS),
            grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(repeat)
    print("Average execution time of kernels:", us, "(us)")

    # Host reference
    with ref_out.map_to_host() as rh:
        for b in range(NBLOCKS):
            chacha_host_block(b, key0, key1, key2, key3, key4, key5, key6, key7,
                              nonce0, nonce1, rh.unsafe_ptr())

    var ok = True
    with d_out.map_to_host() as gh, ref_out.map_to_host() as rh:
        for i in range(NBLOCKS * 16):
            if gh[i] != rh[i]:
                if ok:
                    print("Mismatch at word", i, "gpu=", gh[i], "cpu=", rh[i])
                ok = False
                break
    if ok:
        print("PASS")
    else:
        print("FAIL")
