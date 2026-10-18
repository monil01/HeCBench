using CUDA
using Printf

const BLOCK_SIZE = 256
const C1 = UInt64(0x87c37b91114253d5)
const C2 = UInt64(0x4cf5ad432745937f)

@inline rotl64(x::UInt64, r::UInt32) = (x << r) | (x >> (UInt32(64) - r))

@inline function getblock64(data, base::UInt32, i::UInt32)
    s = UInt64(0)
    p = base + i * UInt32(8)
    for n in UInt32(0):UInt32(7)
        @inbounds s |= UInt64(data[p + n + UInt32(1)]) << (n * UInt32(8))
    end
    return s
end

@inline function fmix64(k0::UInt64)
    k = k0
    k ⊻= k >> UInt32(33)
    k *= UInt64(0xff51afd7ed558ccd)
    k ⊻= k >> UInt32(33)
    k *= UInt64(0xc4ceb9fe1a85ec53)
    k ⊻= k >> UInt32(33)
    return k
end

@inline function murmur_hash(data, base::UInt32, len::UInt32, seed::UInt32)
    nblocks = len ÷ UInt32(16)
    h1 = UInt64(seed)
    h2 = UInt64(seed)

    i = UInt32(0)
    while i < nblocks
        k1 = getblock64(data, base, i * UInt32(2))
        k2 = getblock64(data, base, i * UInt32(2) + UInt32(1))

        k1 *= C1
        k1 = rotl64(k1, UInt32(31))
        k1 *= C2
        h1 ⊻= k1
        h1 = rotl64(h1, UInt32(27))
        h1 += h2
        h1 = h1 * UInt64(5) + UInt64(0x52dce729)

        k2 *= C2
        k2 = rotl64(k2, UInt32(33))
        k2 *= C1
        h2 ⊻= k2
        h2 = rotl64(h2, UInt32(31))
        h2 += h1
        h2 = h2 * UInt64(5) + UInt64(0x38495ab5)
        i += UInt32(1)
    end

    tail = base + nblocks * UInt32(16)
    k1 = UInt64(0)
    k2 = UInt64(0)
    rem = len & UInt32(15)

    if rem >= 15; @inbounds k2 ⊻= UInt64(data[tail + UInt32(15)]) << UInt32(48); end
    if rem >= 14; @inbounds k2 ⊻= UInt64(data[tail + UInt32(14)]) << UInt32(40); end
    if rem >= 13; @inbounds k2 ⊻= UInt64(data[tail + UInt32(13)]) << UInt32(32); end
    if rem >= 12; @inbounds k2 ⊻= UInt64(data[tail + UInt32(12)]) << UInt32(24); end
    if rem >= 11; @inbounds k2 ⊻= UInt64(data[tail + UInt32(11)]) << UInt32(16); end
    if rem >= 10; @inbounds k2 ⊻= UInt64(data[tail + UInt32(10)]) << UInt32(8); end
    if rem >= 9
        @inbounds k2 ⊻= UInt64(data[tail + UInt32(9)])
        k2 *= C2
        k2 = rotl64(k2, UInt32(33))
        k2 *= C1
        h2 ⊻= k2
    end

    if rem >= 8; @inbounds k1 ⊻= UInt64(data[tail + UInt32(8)]) << UInt32(56); end
    if rem >= 7; @inbounds k1 ⊻= UInt64(data[tail + UInt32(7)]) << UInt32(48); end
    if rem >= 6; @inbounds k1 ⊻= UInt64(data[tail + UInt32(6)]) << UInt32(40); end
    if rem >= 5; @inbounds k1 ⊻= UInt64(data[tail + UInt32(5)]) << UInt32(32); end
    if rem >= 4; @inbounds k1 ⊻= UInt64(data[tail + UInt32(4)]) << UInt32(24); end
    if rem >= 3; @inbounds k1 ⊻= UInt64(data[tail + UInt32(3)]) << UInt32(16); end
    if rem >= 2; @inbounds k1 ⊻= UInt64(data[tail + UInt32(2)]) << UInt32(8); end
    if rem >= 1
        @inbounds k1 ⊻= UInt64(data[tail + UInt32(1)])
        k1 *= C1
        k1 = rotl64(k1, UInt32(31))
        k1 *= C2
        h1 ⊻= k1
    end

    h1 ⊻= UInt64(len)
    h2 ⊻= UInt64(len)
    h1 += h2
    h2 += h1
    h1 = fmix64(h1)
    h2 = fmix64(h2)
    h1 += h2
    h2 += h1
    return h1, h2
end

