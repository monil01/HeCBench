# Julia (CUDA.jl) port of the `md5hash` HeCBench benchmark.
#
# Simplified variant matching md5hash-rust / md5hash-triton: sweep the
# fixed keyspace {byteLength=7, valsPerByte=10} (= 10M keys) looking
# for a specific target digest, using the same md5_2words kernel body
# lifted verbatim from the CUDA source's macro expansion.
#
# Usage: julia main.jl <offload> <passes>   (args kept for CLI parity)

using CUDA
using Printf

const BYTE_LEN     = Int32(7)
const VALS_PER_BYTE = Int32(10)
const BLOCK        = 256

# ---- MD5 round helper (device) ----
@inline function md5_round(a::UInt32, b::UInt32, c::UInt32, d::UInt32,
                           w::UInt32, r::UInt32, k::UInt32, f::UInt32)
    a2 = a + f + k + w
    ((a2 << r) | (a2 >> (UInt32(32) - r))) + b
end

# F/G/H/I helpers
@inline mF(x,y,z) = (x & y) | ((~x) & z)
@inline mG(x,y,z) = (x & z) | ((~z) & y)
@inline mH(x,y,z) = x ⊻ y ⊻ z
@inline mI(x,y,z) = y ⊻ (x | (~z))

@inline function md5_2words(w0::UInt32, w1::UInt32, len::UInt32)
    h0 = 0x67452301
    h1 = 0xefcdab89
    h2 = 0x98badcfe
    h3 = 0x10325476
    a, b, c, d = h0, h1, h2, h3
    WL = len * UInt32(8)
    W0 = w0
    W1 = w1
    # padding
    if len == UInt32(0);      W0 |= 0x00000080
    elseif len == UInt32(1);  W0 |= 0x00008000
    elseif len == UInt32(2);  W0 |= 0x00800000
    elseif len == UInt32(3);  W0 |= 0x80000000
    elseif len == UInt32(4);  W1 |= 0x00000080
    elseif len == UInt32(5);  W1 |= 0x00008000
    elseif len == UInt32(6);  W1 |= 0x00800000
    elseif len == UInt32(7);  W1 |= 0x80000000
    end

    # Round 1 — F
    a = md5_round(a,b,c,d, W0, UInt32( 7), UInt32(0xd76aa478), mF(b,c,d)); b_,c_,d_ = a,b,c; d,c,b,a = c_,b_,a,d # rotate a→b→c→d
    d = md5_round(d,a,b,c, W1, UInt32(12), UInt32(0xe8c7b756), mF(a,b,c)); tmp = c; c = b; b = a + ((tmp) - tmp); # placeholder
    # This macro-heavy pattern is ugly in Julia; a cleaner approach follows below.
    return (a, b, c, d)  # placeholder — real body in kernel
end

# Because rewriting all 64 rounds manually with the CUDA macro rotation is
# error-prone, we implement the entire MD5 kernel as an explicit unrolled
# function inside the @cuda kernel below. Each ROUND macro from the CUDA
# source is transliterated as:
#   a = a + f + k + w
#   temp = d; d = c; c = b; b = b + leftrotate(a, r); a = temp
# where leftrotate(v, r) = (v << r) | (v >> (32 - r)).
@inline lrot(v::UInt32, r::UInt32) = (v << r) | (v >> (UInt32(32) - r))

function md5_kernel!(found_idx::CuDeviceVector{Int32},
                    found_key::CuDeviceVector{UInt8},
                    found_dig::CuDeviceVector{UInt32},
                    keyspace::Int32, byte_len::Int32, vals_per_byte::Int32,
                    sd0::UInt32, sd1::UInt32, sd2::UInt32, sd3::UInt32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    startindex = (tid - Int32(1)) * vals_per_byte
    if startindex >= keyspace
        return
    end
    # Compute key = IndexToKey(startindex, byte_len, vals_per_byte)
    key = MVector{8, UInt8}(UInt8(0), UInt8(0), UInt8(0), UInt8(0),
                            UInt8(0), UInt8(0), UInt8(0), UInt8(0))
    idx = startindex
    for i in 1:byte_len
        key[i] = UInt8(idx % vals_per_byte)
        idx ÷= vals_per_byte
    end
    # walk valsPerByte keys, incrementing key[0]
    for j in Int32(0):(vals_per_byte-Int32(1))
        if startindex + j >= keyspace
            break
        end
        # words[0..1] from key bytes
        w0 = UInt32(key[1]) | (UInt32(key[2]) << 8) | (UInt32(key[3]) << 16) | (UInt32(key[4]) << 24)
        w1 = UInt32(key[5]) | (UInt32(key[6]) << 8) | (UInt32(key[7]) << 16) | (UInt32(key[8]) << 24)
        # digest via unrolled MD5
        h0 = UInt32(0x67452301); h1 = UInt32(0xefcdab89)
        h2 = UInt32(0x98badcfe); h3 = UInt32(0x10325476)
        a, b, c, d = h0, h1, h2, h3
        WL = UInt32(byte_len) * UInt32(8)
        W0 = w0; W1 = w1
        # padding (byte_len==7 for this benchmark)
        W1 |= UInt32(0x80000000)
        # Round 1 (F)
        for (w, r, k) in ((W0, UInt32(7),  UInt32(0xd76aa478)),
                          (W1, UInt32(12), UInt32(0xe8c7b756)),
                          (UInt32(0), UInt32(17), UInt32(0x242070db)),
                          (UInt32(0), UInt32(22), UInt32(0xc1bdceee)),
                          (UInt32(0), UInt32(7),  UInt32(0xf57c0faf)),
                          (UInt32(0), UInt32(12), UInt32(0x4787c62a)),
                          (UInt32(0), UInt32(17), UInt32(0xa8304613)),
                          (UInt32(0), UInt32(22), UInt32(0xfd469501)),
                          (UInt32(0), UInt32(7),  UInt32(0x698098d8)),
                          (UInt32(0), UInt32(12), UInt32(0x8b44f7af)),
                          (UInt32(0), UInt32(17), UInt32(0xffff5bb1)),
                          (UInt32(0), UInt32(22), UInt32(0x895cd7be)),
                          (UInt32(0), UInt32(7),  UInt32(0x6b901122)),
                          (UInt32(0), UInt32(12), UInt32(0xfd987193)),
                          (WL,        UInt32(17), UInt32(0xa679438e)),
                          (UInt32(0), UInt32(22), UInt32(0x49b40821)))
            a = a + mF(b,c,d) + k + w
            tmp = d; d = c; c = b; b = b + lrot(a, r); a = tmp
        end
        # Round 2 (G)
        for (w, r, k) in ((W1,        UInt32(5),  UInt32(0xf61e2562)),
                          (UInt32(0), UInt32(9),  UInt32(0xc040b340)),
                          (UInt32(0), UInt32(14), UInt32(0x265e5a51)),
                          (W0,        UInt32(20), UInt32(0xe9b6c7aa)),
                          (UInt32(0), UInt32(5),  UInt32(0xd62f105d)),
                          (UInt32(0), UInt32(9),  UInt32(0x02441453)),
                          (UInt32(0), UInt32(14), UInt32(0xd8a1e681)),
                          (UInt32(0), UInt32(20), UInt32(0xe7d3fbc8)),
                          (UInt32(0), UInt32(5),  UInt32(0x21e1cde6)),
                          (WL,        UInt32(9),  UInt32(0xc33707d6)),
                          (UInt32(0), UInt32(14), UInt32(0xf4d50d87)),
                          (UInt32(0), UInt32(20), UInt32(0x455a14ed)),
                          (UInt32(0), UInt32(5),  UInt32(0xa9e3e905)),
                          (UInt32(0), UInt32(9),  UInt32(0xfcefa3f8)),
                          (UInt32(0), UInt32(14), UInt32(0x676f02d9)),
                          (UInt32(0), UInt32(20), UInt32(0x8d2a4c8a)))
            a = a + mG(b,c,d) + k + w
            tmp = d; d = c; c = b; b = b + lrot(a, r); a = tmp
        end
        # Round 3 (H)
        for (w, r, k) in ((UInt32(0), UInt32(4),  UInt32(0xfffa3942)),
                          (UInt32(0), UInt32(11), UInt32(0x8771f681)),
                          (UInt32(0), UInt32(16), UInt32(0x6d9d6122)),
                          (WL,        UInt32(23), UInt32(0xfde5380c)),
                          (W1,        UInt32(4),  UInt32(0xa4beea44)),
                          (UInt32(0), UInt32(11), UInt32(0x4bdecfa9)),
                          (UInt32(0), UInt32(16), UInt32(0xf6bb4b60)),
                          (UInt32(0), UInt32(23), UInt32(0xbebfbc70)),
                          (UInt32(0), UInt32(4),  UInt32(0x289b7ec6)),
                          (W0,        UInt32(11), UInt32(0xeaa127fa)),
                          (UInt32(0), UInt32(16), UInt32(0xd4ef3085)),
                          (UInt32(0), UInt32(23), UInt32(0x04881d05)),
                          (UInt32(0), UInt32(4),  UInt32(0xd9d4d039)),
                          (UInt32(0), UInt32(11), UInt32(0xe6db99e5)),
                          (UInt32(0), UInt32(16), UInt32(0x1fa27cf8)),
                          (UInt32(0), UInt32(23), UInt32(0xc4ac5665)))
            a = a + mH(b,c,d) + k + w
            tmp = d; d = c; c = b; b = b + lrot(a, r); a = tmp
        end
        # Round 4 (I)
        for (w, r, k) in ((W0,        UInt32(6),  UInt32(0xf4292244)),
                          (UInt32(0), UInt32(10), UInt32(0x432aff97)),
                          (WL,        UInt32(15), UInt32(0xab9423a7)),
                          (UInt32(0), UInt32(21), UInt32(0xfc93a039)),
                          (UInt32(0), UInt32(6),  UInt32(0x655b59c3)),
                          (UInt32(0), UInt32(10), UInt32(0x8f0ccc92)),
                          (UInt32(0), UInt32(15), UInt32(0xffeff47d)),
                          (W1,        UInt32(21), UInt32(0x85845dd1)),
                          (UInt32(0), UInt32(6),  UInt32(0x6fa87e4f)),
                          (UInt32(0), UInt32(10), UInt32(0xfe2ce6e0)),
                          (UInt32(0), UInt32(15), UInt32(0xa3014314)),
                          (UInt32(0), UInt32(21), UInt32(0x4e0811a1)),
                          (UInt32(0), UInt32(6),  UInt32(0xf7537e82)),
                          (UInt32(0), UInt32(10), UInt32(0xbd3af235)),
                          (UInt32(0), UInt32(15), UInt32(0x2ad7d2bb)),
                          (UInt32(0), UInt32(21), UInt32(0xeb86d391)))
            a = a + mI(b,c,d) + k + w
            tmp = d; d = c; c = b; b = b + lrot(a, r); a = tmp
        end
        d0 = h0 + a; d1 = h1 + b; d2 = h2 + c; d3 = h3 + d
        if d0 == sd0 && d1 == sd1 && d2 == sd2 && d3 == sd3
            found_idx[1] = startindex + j
            for k in 1:8; found_key[k] = key[k]; end
            found_dig[1] = d0; found_dig[2] = d1; found_dig[3] = d2; found_dig[4] = d3
        end
        key[1] = key[1] + UInt8(1)
    end
    return
end

# ---- CPU reference (also MD5 unrolled, in bytes) ----
using Base: bswap
function md5_of_key(key::Vector{UInt8})
    # Standard MD5 of 7-byte input using the same md5_2words layout
    w0 = UInt32(key[1]) | (UInt32(key[2]) << 8) | (UInt32(key[3]) << 16) | (UInt32(key[4]) << 24)
    w1 = UInt32(key[5]) | (UInt32(key[6]) << 8) | (UInt32(key[7]) << 16) | (UInt32(0) << 24)
    h0 = UInt32(0x67452301); h1 = UInt32(0xefcdab89)
    h2 = UInt32(0x98badcfe); h3 = UInt32(0x10325476)
    a, b, c, d = h0, h1, h2, h3
    WL = UInt32(7 * 8)
    W0 = w0; W1 = w1 | UInt32(0x80000000)
    lrot_c(v, r) = (v << r) | (v >> (32 - r))
    F(x,y,z) = (x & y) | ((~x) & z)
    G(x,y,z) = (x & z) | ((~z) & y)
    H(x,y,z) = x ⊻ y ⊻ z
    I(x,y,z) = y ⊻ (x | (~z))
    schedule = [
        (W0,7,0xd76aa478,F),(W1,12,0xe8c7b756,F),(UInt32(0),17,0x242070db,F),(UInt32(0),22,0xc1bdceee,F),
        (UInt32(0),7,0xf57c0faf,F),(UInt32(0),12,0x4787c62a,F),(UInt32(0),17,0xa8304613,F),(UInt32(0),22,0xfd469501,F),
        (UInt32(0),7,0x698098d8,F),(UInt32(0),12,0x8b44f7af,F),(UInt32(0),17,0xffff5bb1,F),(UInt32(0),22,0x895cd7be,F),
        (UInt32(0),7,0x6b901122,F),(UInt32(0),12,0xfd987193,F),(WL,17,0xa679438e,F),(UInt32(0),22,0x49b40821,F),
        (W1,5,0xf61e2562,G),(UInt32(0),9,0xc040b340,G),(UInt32(0),14,0x265e5a51,G),(W0,20,0xe9b6c7aa,G),
        (UInt32(0),5,0xd62f105d,G),(UInt32(0),9,0x02441453,G),(UInt32(0),14,0xd8a1e681,G),(UInt32(0),20,0xe7d3fbc8,G),
        (UInt32(0),5,0x21e1cde6,G),(WL,9,0xc33707d6,G),(UInt32(0),14,0xf4d50d87,G),(UInt32(0),20,0x455a14ed,G),
        (UInt32(0),5,0xa9e3e905,G),(UInt32(0),9,0xfcefa3f8,G),(UInt32(0),14,0x676f02d9,G),(UInt32(0),20,0x8d2a4c8a,G),
        (UInt32(0),4,0xfffa3942,H),(UInt32(0),11,0x8771f681,H),(UInt32(0),16,0x6d9d6122,H),(WL,23,0xfde5380c,H),
        (W1,4,0xa4beea44,H),(UInt32(0),11,0x4bdecfa9,H),(UInt32(0),16,0xf6bb4b60,H),(UInt32(0),23,0xbebfbc70,H),
        (UInt32(0),4,0x289b7ec6,H),(W0,11,0xeaa127fa,H),(UInt32(0),16,0xd4ef3085,H),(UInt32(0),23,0x04881d05,H),
        (UInt32(0),4,0xd9d4d039,H),(UInt32(0),11,0xe6db99e5,H),(UInt32(0),16,0x1fa27cf8,H),(UInt32(0),23,0xc4ac5665,H),
        (W0,6,0xf4292244,I),(UInt32(0),10,0x432aff97,I),(WL,15,0xab9423a7,I),(UInt32(0),21,0xfc93a039,I),
        (UInt32(0),6,0x655b59c3,I),(UInt32(0),10,0x8f0ccc92,I),(UInt32(0),15,0xffeff47d,I),(W1,21,0x85845dd1,I),
        (UInt32(0),6,0x6fa87e4f,I),(UInt32(0),10,0xfe2ce6e0,I),(UInt32(0),15,0xa3014314,I),(UInt32(0),21,0x4e0811a1,I),
        (UInt32(0),6,0xf7537e82,I),(UInt32(0),10,0xbd3af235,I),(UInt32(0),15,0x2ad7d2bb,I),(UInt32(0),21,0xeb86d391,I),
    ]
    for (w,r,k,fn) in schedule
        a = a + fn(b,c,d) + UInt32(k) + w
        tmp = d; d = c; c = b; b = b + UInt32(lrot_c(a, r)); a = tmp
    end
    (h0 + a, h1 + b, h2 + c, h3 + d)
end

function index_to_key(index::Int32, byte_len::Int32, vals_per_byte::Int32)
    key = zeros(UInt8, 8)
    idx = index
    for i in 1:byte_len
        key[i] = UInt8(idx % vals_per_byte)
        idx ÷= vals_per_byte
    end
    key
end

function keyspace_size(byte_len::Int32, vals_per_byte::Int32)
    k::Int32 = Int32(1)
    for _ in 1:byte_len
        k *= vals_per_byte
    end
    k
end

using StaticArrays

function main()
    passes = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    keyspace = keyspace_size(BYTE_LEN, VALS_PER_BYTE)
    @printf("MD5 keyspace: %d (byteLength=7, valsPerByte=10)\n", keyspace)

    target_index::Int32 = keyspace ÷ Int32(3) + Int32(42)
    target_key = index_to_key(target_index, BYTE_LEN, VALS_PER_BYTE)
    (t0, t1, t2, t3) = md5_of_key(target_key)
    @printf("Target index %d, digest %08x%08x%08x%08x\n",
            target_index, t0, t1, t2, t3)

    d_found_idx = CUDA.fill(Int32(-1), 1)
    d_found_key = CUDA.zeros(UInt8, 8)
    d_found_dig = CUDA.zeros(UInt32, 4)

    n_threads = cld(keyspace, VALS_PER_BYTE)
    blocks = cld(n_threads, BLOCK)

    CUDA.synchronize()
    t0_wall = time_ns()
    for _ in 1:passes
        @cuda threads=BLOCK blocks=blocks md5_kernel!(
            d_found_idx, d_found_key, d_found_dig,
            keyspace, BYTE_LEN, VALS_PER_BYTE, t0, t1, t2, t3)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0_wall) / 1e9
    @printf("Passes: %d, keyspace %d => %d hashes in %.2f s = %.2f M/s\n",
            passes, keyspace, keyspace * passes, elapsed_s,
            (keyspace * passes) / elapsed_s / 1e6)

    fi = Array(d_found_idx)
    fk = Array(d_found_key)
    fd = Array(d_found_dig)
    @printf("Found: index=%d, key=%s, digest[0]=%08x\n", fi[1], string(fk[1:7]), fd[1])

    ok = fi[1] == target_index && fk[1:7] == target_key[1:7] &&
         fd[1] == t0 && fd[2] == t1 && fd[3] == t2 && fd[4] == t3
    println(ok ? "PASS" : "FAIL")
end

main()
