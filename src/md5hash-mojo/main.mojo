# Mojo GPU port of the `md5hash` HeCBench benchmark (simplified,
# byteLength=7, valsPerByte=10 fixed to match md5hash-{rust,julia,triton}).
#
# Each GPU thread walks `valsPerByte` keys, computes their MD5 via the
# full 64-round algorithm, and records the index+key+digest when it
# matches a searched-for digest. No atomics — a single flag slot is
# written by whichever thread finds it. Verified against a host-side
# Mojo MD5 computed for the target key.
#
# Usage: main.mojo <offload> <passes>

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


alias BYTE_LEN: Int32 = 7
alias VALS_PER_BYTE: Int32 = 10


@always_inline
def lrot(v: UInt32, r: UInt32) -> UInt32:
    return (v << r) | (v >> (UInt32(32) - r))

@always_inline
def F(x: UInt32, y: UInt32, z: UInt32) -> UInt32: return (x & y) | ((~x) & z)
@always_inline
def G(x: UInt32, y: UInt32, z: UInt32) -> UInt32: return (x & z) | ((~z) & y)
@always_inline
def H(x: UInt32, y: UInt32, z: UInt32) -> UInt32: return x ^ y ^ z
@always_inline
def I(x: UInt32, y: UInt32, z: UInt32) -> UInt32: return y ^ (x | (~z))


def md5_kernel(
        found_idx: UnsafePointer[Int32, MutAnyOrigin],
        found_key: UnsafePointer[UInt8, MutAnyOrigin],
        found_dig: UnsafePointer[UInt32, MutAnyOrigin],
        keyspace: Int32,
        sd0: UInt32, sd1: UInt32, sd2: UInt32, sd3: UInt32):
    var tid = Int32(block_idx.x * block_dim.x + thread_idx.x)
    var startindex = tid * VALS_PER_BYTE
    if startindex >= keyspace:
        return
    # IndexToKey — byte_len=7 unrolled
    var k0: UInt8 = UInt8(startindex % VALS_PER_BYTE)
    var idx = startindex // VALS_PER_BYTE
    var k1: UInt8 = UInt8(idx % VALS_PER_BYTE); idx = idx // VALS_PER_BYTE
    var k2: UInt8 = UInt8(idx % VALS_PER_BYTE); idx = idx // VALS_PER_BYTE
    var k3: UInt8 = UInt8(idx % VALS_PER_BYTE); idx = idx // VALS_PER_BYTE
    var k4: UInt8 = UInt8(idx % VALS_PER_BYTE); idx = idx // VALS_PER_BYTE
    var k5: UInt8 = UInt8(idx % VALS_PER_BYTE); idx = idx // VALS_PER_BYTE
    var k6: UInt8 = UInt8(idx % VALS_PER_BYTE)

    for j in range(VALS_PER_BYTE):
        if startindex + j >= keyspace:
            break
        var W0: UInt32 = UInt32(k0) | (UInt32(k1) << UInt32(8)) | (UInt32(k2) << UInt32(16)) | (UInt32(k3) << UInt32(24))
        var W1: UInt32 = UInt32(k4) | (UInt32(k5) << UInt32(8)) | (UInt32(k6) << UInt32(16))
        # padding: byte_len=7 → W1 |= 0x80000000
        W1 = W1 | UInt32(0x80000000)
        var WL: UInt32 = UInt32(7 * 8)

        var h0: UInt32 = UInt32(0x67452301)
        var h1: UInt32 = UInt32(0xefcdab89)
        var h2: UInt32 = UInt32(0x98badcfe)
        var h3: UInt32 = UInt32(0x10325476)
        var a = h0; var b = h1; var c = h2; var d = h3

        # ROUND1 (F) — 16 iterations
        # I'll unroll the entire schedule as a straight list to avoid list literals.
        # Each: a = a + F(b,c,d) + k + w; tmp=d; d=c; c=b; b=b+lrot(a,r); a=tmp
        # Round 1
        a = a + F(b,c,d) + UInt32(0xd76aa478) + W0; var tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
        a = a + F(b,c,d) + UInt32(0xe8c7b756) + W1; tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
        a = a + F(b,c,d) + UInt32(0x242070db) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
        a = a + F(b,c,d) + UInt32(0xc1bdceee) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
        a = a + F(b,c,d) + UInt32(0xf57c0faf) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
        a = a + F(b,c,d) + UInt32(0x4787c62a) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
        a = a + F(b,c,d) + UInt32(0xa8304613) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
        a = a + F(b,c,d) + UInt32(0xfd469501) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
        a = a + F(b,c,d) + UInt32(0x698098d8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
        a = a + F(b,c,d) + UInt32(0x8b44f7af) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
        a = a + F(b,c,d) + UInt32(0xffff5bb1) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
        a = a + F(b,c,d) + UInt32(0x895cd7be) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
        a = a + F(b,c,d) + UInt32(0x6b901122) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
        a = a + F(b,c,d) + UInt32(0xfd987193) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
        a = a + F(b,c,d) + UInt32(0xa679438e) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
        a = a + F(b,c,d) + UInt32(0x49b40821) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
        # Round 2 (G)
        a = a + G(b,c,d) + UInt32(0xf61e2562) + W1;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
        a = a + G(b,c,d) + UInt32(0xc040b340) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
        a = a + G(b,c,d) + UInt32(0x265e5a51) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
        a = a + G(b,c,d) + UInt32(0xe9b6c7aa) + W0;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
        a = a + G(b,c,d) + UInt32(0xd62f105d) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
        a = a + G(b,c,d) + UInt32(0x02441453) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
        a = a + G(b,c,d) + UInt32(0xd8a1e681) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
        a = a + G(b,c,d) + UInt32(0xe7d3fbc8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
        a = a + G(b,c,d) + UInt32(0x21e1cde6) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
        a = a + G(b,c,d) + UInt32(0xc33707d6) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
        a = a + G(b,c,d) + UInt32(0xf4d50d87) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
        a = a + G(b,c,d) + UInt32(0x455a14ed) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
        a = a + G(b,c,d) + UInt32(0xa9e3e905) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
        a = a + G(b,c,d) + UInt32(0xfcefa3f8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
        a = a + G(b,c,d) + UInt32(0x676f02d9) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
        a = a + G(b,c,d) + UInt32(0x8d2a4c8a) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
        # Round 3 (H)
        a = a + H(b,c,d) + UInt32(0xfffa3942) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
        a = a + H(b,c,d) + UInt32(0x8771f681) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
        a = a + H(b,c,d) + UInt32(0x6d9d6122) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
        a = a + H(b,c,d) + UInt32(0xfde5380c) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
        a = a + H(b,c,d) + UInt32(0xa4beea44) + W1;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
        a = a + H(b,c,d) + UInt32(0x4bdecfa9) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
        a = a + H(b,c,d) + UInt32(0xf6bb4b60) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
        a = a + H(b,c,d) + UInt32(0xbebfbc70) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
        a = a + H(b,c,d) + UInt32(0x289b7ec6) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
        a = a + H(b,c,d) + UInt32(0xeaa127fa) + W0;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
        a = a + H(b,c,d) + UInt32(0xd4ef3085) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
        a = a + H(b,c,d) + UInt32(0x04881d05) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
        a = a + H(b,c,d) + UInt32(0xd9d4d039) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
        a = a + H(b,c,d) + UInt32(0xe6db99e5) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
        a = a + H(b,c,d) + UInt32(0x1fa27cf8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
        a = a + H(b,c,d) + UInt32(0xc4ac5665) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
        # Round 4 (I)
        a = a + I(b,c,d) + UInt32(0xf4292244) + W0;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
        a = a + I(b,c,d) + UInt32(0x432aff97) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
        a = a + I(b,c,d) + UInt32(0xab9423a7) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
        a = a + I(b,c,d) + UInt32(0xfc93a039) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
        a = a + I(b,c,d) + UInt32(0x655b59c3) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
        a = a + I(b,c,d) + UInt32(0x8f0ccc92) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
        a = a + I(b,c,d) + UInt32(0xffeff47d) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
        a = a + I(b,c,d) + UInt32(0x85845dd1) + W1;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
        a = a + I(b,c,d) + UInt32(0x6fa87e4f) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
        a = a + I(b,c,d) + UInt32(0xfe2ce6e0) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
        a = a + I(b,c,d) + UInt32(0xa3014314) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
        a = a + I(b,c,d) + UInt32(0x4e0811a1) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
        a = a + I(b,c,d) + UInt32(0xf7537e82) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
        a = a + I(b,c,d) + UInt32(0xbd3af235) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
        a = a + I(b,c,d) + UInt32(0x2ad7d2bb) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
        a = a + I(b,c,d) + UInt32(0xeb86d391) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp

        var d0 = h0 + a
        var d1 = h1 + b
        var d2 = h2 + c
        var d3 = h3 + d
        if d0 == sd0 and d1 == sd1 and d2 == sd2 and d3 == sd3:
            found_idx[0] = startindex + j
            found_key[0] = k0
            found_key[1] = k1
            found_key[2] = k2
            found_key[3] = k3
            found_key[4] = k4
            found_key[5] = k5
            found_key[6] = k6
            found_dig[0] = d0
            found_dig[1] = d1
            found_dig[2] = d2
            found_dig[3] = d3
        k0 = k0 + UInt8(1)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage:", args[0], "<offload> <passes>")
        exit(1)
    var passes = Int(atol(args[2])) if len(args) >= 3 else 1

    # Fixed sweep: byte_len=7, vals_per_byte=10 → keyspace = 10^7
    var keyspace: Int32 = Int32(10_000_000)
    print("MD5 keyspace:", keyspace, "(byteLength=7, valsPerByte=10)")

    # Pick a target index deterministically
    var target_idx: Int32 = keyspace // Int32(3) + Int32(42)
    var target_key = List[UInt8](capacity=8)
    for _ in range(8): target_key.append(UInt8(0))
    var idx = target_idx
    for i in range(7):
        target_key[i] = UInt8(idx % VALS_PER_BYTE)
        idx = idx // VALS_PER_BYTE
    # Compute target digest via a single-thread invocation of the same
    # algorithm on the host: cheapest is to just launch the kernel and read
    # the found value. But we need the search digest first!
    # Do a single-host-side MD5 in the same layout.
    var W0: UInt32 = UInt32(target_key[0]) | (UInt32(target_key[1]) << UInt32(8)) | (UInt32(target_key[2]) << UInt32(16)) | (UInt32(target_key[3]) << UInt32(24))
    var W1: UInt32 = UInt32(target_key[4]) | (UInt32(target_key[5]) << UInt32(8)) | (UInt32(target_key[6]) << UInt32(16))
    W1 = W1 | UInt32(0x80000000)
    var WL: UInt32 = UInt32(7 * 8)
    var h0: UInt32 = UInt32(0x67452301)
    var h1: UInt32 = UInt32(0xefcdab89)
    var h2: UInt32 = UInt32(0x98badcfe)
    var h3: UInt32 = UInt32(0x10325476)
    var a = h0; var b = h1; var c = h2; var d = h3
    a = a + F(b,c,d) + UInt32(0xd76aa478) + W0; var tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
    a = a + F(b,c,d) + UInt32(0xe8c7b756) + W1; tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
    a = a + F(b,c,d) + UInt32(0x242070db) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
    a = a + F(b,c,d) + UInt32(0xc1bdceee) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
    a = a + F(b,c,d) + UInt32(0xf57c0faf) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
    a = a + F(b,c,d) + UInt32(0x4787c62a) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
    a = a + F(b,c,d) + UInt32(0xa8304613) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
    a = a + F(b,c,d) + UInt32(0xfd469501) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
    a = a + F(b,c,d) + UInt32(0x698098d8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
    a = a + F(b,c,d) + UInt32(0x8b44f7af) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
    a = a + F(b,c,d) + UInt32(0xffff5bb1) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
    a = a + F(b,c,d) + UInt32(0x895cd7be) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
    a = a + F(b,c,d) + UInt32(0x6b901122) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(7)); a = tmp
    a = a + F(b,c,d) + UInt32(0xfd987193) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(12)); a = tmp
    a = a + F(b,c,d) + UInt32(0xa679438e) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(17)); a = tmp
    a = a + F(b,c,d) + UInt32(0x49b40821) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(22)); a = tmp
    a = a + G(b,c,d) + UInt32(0xf61e2562) + W1;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
    a = a + G(b,c,d) + UInt32(0xc040b340) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
    a = a + G(b,c,d) + UInt32(0x265e5a51) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
    a = a + G(b,c,d) + UInt32(0xe9b6c7aa) + W0;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
    a = a + G(b,c,d) + UInt32(0xd62f105d) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
    a = a + G(b,c,d) + UInt32(0x02441453) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
    a = a + G(b,c,d) + UInt32(0xd8a1e681) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
    a = a + G(b,c,d) + UInt32(0xe7d3fbc8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
    a = a + G(b,c,d) + UInt32(0x21e1cde6) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
    a = a + G(b,c,d) + UInt32(0xc33707d6) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
    a = a + G(b,c,d) + UInt32(0xf4d50d87) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
    a = a + G(b,c,d) + UInt32(0x455a14ed) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
    a = a + G(b,c,d) + UInt32(0xa9e3e905) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(5)); a = tmp
    a = a + G(b,c,d) + UInt32(0xfcefa3f8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(9)); a = tmp
    a = a + G(b,c,d) + UInt32(0x676f02d9) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(14)); a = tmp
    a = a + G(b,c,d) + UInt32(0x8d2a4c8a) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(20)); a = tmp
    a = a + H(b,c,d) + UInt32(0xfffa3942) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
    a = a + H(b,c,d) + UInt32(0x8771f681) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
    a = a + H(b,c,d) + UInt32(0x6d9d6122) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
    a = a + H(b,c,d) + UInt32(0xfde5380c) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
    a = a + H(b,c,d) + UInt32(0xa4beea44) + W1;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
    a = a + H(b,c,d) + UInt32(0x4bdecfa9) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
    a = a + H(b,c,d) + UInt32(0xf6bb4b60) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
    a = a + H(b,c,d) + UInt32(0xbebfbc70) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
    a = a + H(b,c,d) + UInt32(0x289b7ec6) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
    a = a + H(b,c,d) + UInt32(0xeaa127fa) + W0;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
    a = a + H(b,c,d) + UInt32(0xd4ef3085) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
    a = a + H(b,c,d) + UInt32(0x04881d05) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
    a = a + H(b,c,d) + UInt32(0xd9d4d039) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(4)); a = tmp
    a = a + H(b,c,d) + UInt32(0xe6db99e5) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(11)); a = tmp
    a = a + H(b,c,d) + UInt32(0x1fa27cf8) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(16)); a = tmp
    a = a + H(b,c,d) + UInt32(0xc4ac5665) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(23)); a = tmp
    a = a + I(b,c,d) + UInt32(0xf4292244) + W0;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
    a = a + I(b,c,d) + UInt32(0x432aff97) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
    a = a + I(b,c,d) + UInt32(0xab9423a7) + WL;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
    a = a + I(b,c,d) + UInt32(0xfc93a039) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
    a = a + I(b,c,d) + UInt32(0x655b59c3) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
    a = a + I(b,c,d) + UInt32(0x8f0ccc92) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
    a = a + I(b,c,d) + UInt32(0xffeff47d) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
    a = a + I(b,c,d) + UInt32(0x85845dd1) + W1;         tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
    a = a + I(b,c,d) + UInt32(0x6fa87e4f) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
    a = a + I(b,c,d) + UInt32(0xfe2ce6e0) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
    a = a + I(b,c,d) + UInt32(0xa3014314) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
    a = a + I(b,c,d) + UInt32(0x4e0811a1) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
    a = a + I(b,c,d) + UInt32(0xf7537e82) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(6)); a = tmp
    a = a + I(b,c,d) + UInt32(0xbd3af235) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(10)); a = tmp
    a = a + I(b,c,d) + UInt32(0x2ad7d2bb) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(15)); a = tmp
    a = a + I(b,c,d) + UInt32(0xeb86d391) + UInt32(0); tmp = d; d = c; c = b; b = b + lrot(a, UInt32(21)); a = tmp
    var t0 = h0 + a
    var t1 = h1 + b
    var t2 = h2 + c
    var t3 = h3 + d
    print("Target index", target_idx, "digest", hex(t0), hex(t1), hex(t2), hex(t3))

    var ctx = DeviceContext()
    var d_found_idx = ctx.enqueue_create_buffer[DType.int32](1)
    var d_found_key = ctx.enqueue_create_buffer[DType.uint8](8)
    var d_found_dig = ctx.enqueue_create_buffer[DType.uint32](4)
    with d_found_idx.map_to_host() as h: h[0] = Int32(-1)

    comptime BLOCK: Int = 256
    var n_threads = (keyspace + VALS_PER_BYTE - 1) // VALS_PER_BYTE
    var grid = (Int(n_threads) + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var ts0 = perf_counter_ns()
    for _ in range(passes):
        ctx.enqueue_function[func=md5_kernel](
            d_found_idx.unsafe_ptr(), d_found_key.unsafe_ptr(), d_found_dig.unsafe_ptr(),
            keyspace, t0, t1, t2, t3,
            grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    var elapsed_s = Float64(perf_counter_ns() - ts0) / 1e9
    var mhps = Float64(Int(keyspace) * passes) / elapsed_s / 1e6
    print("Passes:", passes, "keyspace", keyspace, "=>", Int(keyspace) * passes,
          "hashes in", elapsed_s, "s =", mhps, "M/s")

    var ok = True
    with d_found_idx.map_to_host() as h:
        if h[0] != target_idx:
            ok = False
            print("FAIL: expected idx", target_idx, "got", h[0])
    if ok:
        print("PASS")
