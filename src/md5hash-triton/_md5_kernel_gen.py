import triton
import triton.language as tl

@triton.jit
def md5_kernel(
    found_idx_ptr, found_key_ptr, found_dig_ptr,
    keyspace, valsPerByte,
    sd0, sd1, sd2, sd3,
    BLOCK: tl.constexpr, VPB: tl.constexpr, BLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    start_idx = tid * valsPerByte
    idx = start_idx
    v0 = idx % valsPerByte; idx = idx // valsPerByte
    v1 = idx % valsPerByte; idx = idx // valsPerByte
    v2 = idx % valsPerByte; idx = idx // valsPerByte
    v3 = idx % valsPerByte; idx = idx // valsPerByte
    v4 = idx % valsPerByte; idx = idx // valsPerByte
    v5 = idx % valsPerByte; idx = idx // valsPerByte
    v6 = idx % valsPerByte; idx = idx // valsPerByte
    v7 = idx % valsPerByte
    v0 = v0.to(tl.uint32); v1 = v1.to(tl.uint32); v2 = v2.to(tl.uint32); v3 = v3.to(tl.uint32)
    v4 = v4.to(tl.uint32); v5 = v5.to(tl.uint32); v6 = v6.to(tl.uint32); v7 = v7.to(tl.uint32)
    for j in tl.static_range(VPB):
        active = (start_idx + j) < keyspace
        W0 = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)
        W1 = v4 | (v5 << 8) | (v6 << 16) | (v7 << 24)
        W1 = W1 | 0x80000000
        WL = tl.full(W0.shape, BLEN * 8, tl.uint32)
        zero = tl.full(W0.shape, 0, tl.uint32)
        a = tl.full(W0.shape, 0x67452301, tl.uint32)
        b = tl.full(W0.shape, 0xefcdab89, tl.uint32)
        c = tl.full(W0.shape, 0x98badcfe, tl.uint32)
        d = tl.full(W0.shape, 0x10325476, tl.uint32)
        a = a + ((b & c) | ((~b) & d)) + W0 + tl.full(W0.shape, 0xd76aa478, tl.uint32)
        a = (a << 7) | (a >> 25)
        a = b + a
        d = d + ((a & b) | ((~a) & c)) + W1 + tl.full(W0.shape, 0xe8c7b756, tl.uint32)
        d = (d << 12) | (d >> 20)
        d = a + d
        c = c + ((d & a) | ((~d) & b)) + zero + tl.full(W0.shape, 0x242070db, tl.uint32)
        c = (c << 17) | (c >> 15)
        c = d + c
        b = b + ((c & d) | ((~c) & a)) + zero + tl.full(W0.shape, 0xc1bdceee, tl.uint32)
        b = (b << 22) | (b >> 10)
        b = c + b
        a = a + ((b & c) | ((~b) & d)) + zero + tl.full(W0.shape, 0xf57c0faf, tl.uint32)
        a = (a << 7) | (a >> 25)
        a = b + a
        d = d + ((a & b) | ((~a) & c)) + zero + tl.full(W0.shape, 0x4787c62a, tl.uint32)
        d = (d << 12) | (d >> 20)
        d = a + d
        c = c + ((d & a) | ((~d) & b)) + zero + tl.full(W0.shape, 0xa8304613, tl.uint32)
        c = (c << 17) | (c >> 15)
        c = d + c
        b = b + ((c & d) | ((~c) & a)) + zero + tl.full(W0.shape, 0xfd469501, tl.uint32)
        b = (b << 22) | (b >> 10)
        b = c + b
        a = a + ((b & c) | ((~b) & d)) + zero + tl.full(W0.shape, 0x698098d8, tl.uint32)
        a = (a << 7) | (a >> 25)
        a = b + a
        d = d + ((a & b) | ((~a) & c)) + zero + tl.full(W0.shape, 0x8b44f7af, tl.uint32)
        d = (d << 12) | (d >> 20)
        d = a + d
        c = c + ((d & a) | ((~d) & b)) + zero + tl.full(W0.shape, 0xffff5bb1, tl.uint32)
        c = (c << 17) | (c >> 15)
        c = d + c
        b = b + ((c & d) | ((~c) & a)) + zero + tl.full(W0.shape, 0x895cd7be, tl.uint32)
        b = (b << 22) | (b >> 10)
        b = c + b
        a = a + ((b & c) | ((~b) & d)) + zero + tl.full(W0.shape, 0x6b901122, tl.uint32)
        a = (a << 7) | (a >> 25)
        a = b + a
        d = d + ((a & b) | ((~a) & c)) + zero + tl.full(W0.shape, 0xfd987193, tl.uint32)
        d = (d << 12) | (d >> 20)
        d = a + d
        c = c + ((d & a) | ((~d) & b)) + WL + tl.full(W0.shape, 0xa679438e, tl.uint32)
        c = (c << 17) | (c >> 15)
        c = d + c
        b = b + ((c & d) | ((~c) & a)) + zero + tl.full(W0.shape, 0x49b40821, tl.uint32)
        b = (b << 22) | (b >> 10)
        b = c + b
        a = a + ((b & d) | ((~d) & c)) + W1 + tl.full(W0.shape, 0xf61e2562, tl.uint32)
        a = (a << 5) | (a >> 27)
        a = b + a
        d = d + ((a & c) | ((~c) & b)) + zero + tl.full(W0.shape, 0xc040b340, tl.uint32)
        d = (d << 9) | (d >> 23)
        d = a + d
        c = c + ((d & b) | ((~b) & a)) + zero + tl.full(W0.shape, 0x265e5a51, tl.uint32)
        c = (c << 14) | (c >> 18)
        c = d + c
        b = b + ((c & a) | ((~a) & d)) + W0 + tl.full(W0.shape, 0xe9b6c7aa, tl.uint32)
        b = (b << 20) | (b >> 12)
        b = c + b
        a = a + ((b & d) | ((~d) & c)) + zero + tl.full(W0.shape, 0xd62f105d, tl.uint32)
        a = (a << 5) | (a >> 27)
        a = b + a
        d = d + ((a & c) | ((~c) & b)) + zero + tl.full(W0.shape, 0x02441453, tl.uint32)
        d = (d << 9) | (d >> 23)
        d = a + d
        c = c + ((d & b) | ((~b) & a)) + zero + tl.full(W0.shape, 0xd8a1e681, tl.uint32)
        c = (c << 14) | (c >> 18)
        c = d + c
        b = b + ((c & a) | ((~a) & d)) + zero + tl.full(W0.shape, 0xe7d3fbc8, tl.uint32)
        b = (b << 20) | (b >> 12)
        b = c + b
        a = a + ((b & d) | ((~d) & c)) + zero + tl.full(W0.shape, 0x21e1cde6, tl.uint32)
        a = (a << 5) | (a >> 27)
        a = b + a
        d = d + ((a & c) | ((~c) & b)) + WL + tl.full(W0.shape, 0xc33707d6, tl.uint32)
        d = (d << 9) | (d >> 23)
        d = a + d
        c = c + ((d & b) | ((~b) & a)) + zero + tl.full(W0.shape, 0xf4d50d87, tl.uint32)
        c = (c << 14) | (c >> 18)
        c = d + c
        b = b + ((c & a) | ((~a) & d)) + zero + tl.full(W0.shape, 0x455a14ed, tl.uint32)
        b = (b << 20) | (b >> 12)
        b = c + b
        a = a + ((b & d) | ((~d) & c)) + zero + tl.full(W0.shape, 0xa9e3e905, tl.uint32)
        a = (a << 5) | (a >> 27)
        a = b + a
        d = d + ((a & c) | ((~c) & b)) + zero + tl.full(W0.shape, 0xfcefa3f8, tl.uint32)
        d = (d << 9) | (d >> 23)
        d = a + d
        c = c + ((d & b) | ((~b) & a)) + zero + tl.full(W0.shape, 0x676f02d9, tl.uint32)
        c = (c << 14) | (c >> 18)
        c = d + c
        b = b + ((c & a) | ((~a) & d)) + zero + tl.full(W0.shape, 0x8d2a4c8a, tl.uint32)
        b = (b << 20) | (b >> 12)
        b = c + b
        a = a + (b ^ c ^ d) + zero + tl.full(W0.shape, 0xfffa3942, tl.uint32)
        a = (a << 4) | (a >> 28)
        a = b + a
        d = d + (a ^ b ^ c) + zero + tl.full(W0.shape, 0x8771f681, tl.uint32)
        d = (d << 11) | (d >> 21)
        d = a + d
        c = c + (d ^ a ^ b) + zero + tl.full(W0.shape, 0x6d9d6122, tl.uint32)
        c = (c << 16) | (c >> 16)
        c = d + c
        b = b + (c ^ d ^ a) + WL + tl.full(W0.shape, 0xfde5380c, tl.uint32)
        b = (b << 23) | (b >> 9)
        b = c + b
        a = a + (b ^ c ^ d) + W1 + tl.full(W0.shape, 0xa4beea44, tl.uint32)
        a = (a << 4) | (a >> 28)
        a = b + a
        d = d + (a ^ b ^ c) + zero + tl.full(W0.shape, 0x4bdecfa9, tl.uint32)
        d = (d << 11) | (d >> 21)
        d = a + d
        c = c + (d ^ a ^ b) + zero + tl.full(W0.shape, 0xf6bb4b60, tl.uint32)
        c = (c << 16) | (c >> 16)
        c = d + c
        b = b + (c ^ d ^ a) + zero + tl.full(W0.shape, 0xbebfbc70, tl.uint32)
        b = (b << 23) | (b >> 9)
        b = c + b
        a = a + (b ^ c ^ d) + zero + tl.full(W0.shape, 0x289b7ec6, tl.uint32)
        a = (a << 4) | (a >> 28)
        a = b + a
        d = d + (a ^ b ^ c) + W0 + tl.full(W0.shape, 0xeaa127fa, tl.uint32)
        d = (d << 11) | (d >> 21)
        d = a + d
        c = c + (d ^ a ^ b) + zero + tl.full(W0.shape, 0xd4ef3085, tl.uint32)
        c = (c << 16) | (c >> 16)
        c = d + c
        b = b + (c ^ d ^ a) + zero + tl.full(W0.shape, 0x04881d05, tl.uint32)
        b = (b << 23) | (b >> 9)
        b = c + b
        a = a + (b ^ c ^ d) + zero + tl.full(W0.shape, 0xd9d4d039, tl.uint32)
        a = (a << 4) | (a >> 28)
        a = b + a
        d = d + (a ^ b ^ c) + zero + tl.full(W0.shape, 0xe6db99e5, tl.uint32)
        d = (d << 11) | (d >> 21)
        d = a + d
        c = c + (d ^ a ^ b) + zero + tl.full(W0.shape, 0x1fa27cf8, tl.uint32)
        c = (c << 16) | (c >> 16)
        c = d + c
        b = b + (c ^ d ^ a) + zero + tl.full(W0.shape, 0xc4ac5665, tl.uint32)
        b = (b << 23) | (b >> 9)
        b = c + b
        a = a + (c ^ (b | (~d))) + W0 + tl.full(W0.shape, 0xf4292244, tl.uint32)
        a = (a << 6) | (a >> 26)
        a = b + a
        d = d + (b ^ (a | (~c))) + zero + tl.full(W0.shape, 0x432aff97, tl.uint32)
        d = (d << 10) | (d >> 22)
        d = a + d
        c = c + (a ^ (d | (~b))) + WL + tl.full(W0.shape, 0xab9423a7, tl.uint32)
        c = (c << 15) | (c >> 17)
        c = d + c
        b = b + (d ^ (c | (~a))) + zero + tl.full(W0.shape, 0xfc93a039, tl.uint32)
        b = (b << 21) | (b >> 11)
        b = c + b
        a = a + (c ^ (b | (~d))) + zero + tl.full(W0.shape, 0x655b59c3, tl.uint32)
        a = (a << 6) | (a >> 26)
        a = b + a
        d = d + (b ^ (a | (~c))) + zero + tl.full(W0.shape, 0x8f0ccc92, tl.uint32)
        d = (d << 10) | (d >> 22)
        d = a + d
        c = c + (a ^ (d | (~b))) + zero + tl.full(W0.shape, 0xffeff47d, tl.uint32)
        c = (c << 15) | (c >> 17)
        c = d + c
        b = b + (d ^ (c | (~a))) + W1 + tl.full(W0.shape, 0x85845dd1, tl.uint32)
        b = (b << 21) | (b >> 11)
        b = c + b
        a = a + (c ^ (b | (~d))) + zero + tl.full(W0.shape, 0x6fa87e4f, tl.uint32)
        a = (a << 6) | (a >> 26)
        a = b + a
        d = d + (b ^ (a | (~c))) + zero + tl.full(W0.shape, 0xfe2ce6e0, tl.uint32)
        d = (d << 10) | (d >> 22)
        d = a + d
        c = c + (a ^ (d | (~b))) + zero + tl.full(W0.shape, 0xa3014314, tl.uint32)
        c = (c << 15) | (c >> 17)
        c = d + c
        b = b + (d ^ (c | (~a))) + zero + tl.full(W0.shape, 0x4e0811a1, tl.uint32)
        b = (b << 21) | (b >> 11)
        b = c + b
        a = a + (c ^ (b | (~d))) + zero + tl.full(W0.shape, 0xf7537e82, tl.uint32)
        a = (a << 6) | (a >> 26)
        a = b + a
        d = d + (b ^ (a | (~c))) + zero + tl.full(W0.shape, 0xbd3af235, tl.uint32)
        d = (d << 10) | (d >> 22)
        d = a + d
        c = c + (a ^ (d | (~b))) + zero + tl.full(W0.shape, 0x2ad7d2bb, tl.uint32)
        c = (c << 15) | (c >> 17)
        c = d + c
        b = b + (d ^ (c | (~a))) + zero + tl.full(W0.shape, 0xeb86d391, tl.uint32)
        b = (b << 21) | (b >> 11)
        b = c + b
        h0 = a + tl.full(a.shape, 0x67452301, tl.uint32)
        h1 = b + tl.full(a.shape, 0xefcdab89, tl.uint32)
        h2 = c + tl.full(a.shape, 0x98badcfe, tl.uint32)
        h3 = d + tl.full(a.shape, 0x10325476, tl.uint32)
        match = (h0 == sd0) & (h1 == sd1) & (h2 == sd2) & (h3 == sd3) & active
        idx_val = (start_idx + j).to(tl.int32)
        big = tl.full(idx_val.shape, 0x7fffffff, tl.int32)
        idx_broadcast = tl.where(match, idx_val, big)
        min_idx = tl.min(idx_broadcast, axis=0)
        tl.atomic_min(found_idx_ptr, min_idx)
        v0 = ((v0 + tl.full(v0.shape, 1, tl.uint32)) % valsPerByte.to(tl.uint32))
