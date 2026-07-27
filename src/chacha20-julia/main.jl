using CUDA
using Printf
using StaticArrays

# Julia port of chacha20-cuda benchmark.
# One kernel: generates keystream (equivalent to XOR with zero plaintext) and
# writes it to the result buffer. Parallelized per 64-byte block.

@inline function rotl32(x::UInt32, n::Int32)
    return (x << n) | (x >> (Int32(32) - n))
end

@inline function pack4_le(a0::UInt8, a1::UInt8, a2::UInt8, a3::UInt8)
    return UInt32(a0) | (UInt32(a1) << 8) | (UInt32(a2) << 16) | (UInt32(a3) << 24)
end

# Cooperative kernel: threadIdx maps to one 64-byte block.
function chacha_kernel!(key_words, nonce_words, out_bytes, num_blocks::Int32)
    tid = Int32(threadIdx().x)
    if tid > num_blocks
        return
    end
    counter = UInt64(tid - Int32(1))  # block index 0..num_blocks-1

    # state[1..16]
    s = @MArray zeros(UInt32, 16)
    # magic constant "expand 32-byte k" packed little-endian
    s[1]  = UInt32(0x61707865)
    s[2]  = UInt32(0x3320646e)
    s[3]  = UInt32(0x79622d32)
    s[4]  = UInt32(0x6b206574)
    @inbounds for i in 1:8
        s[i + 4] = key_words[i]
    end
    s[13] = UInt32(counter & UInt64(0xFFFFFFFF))
    s[14] = UInt32((counter >> 32) & UInt64(0xFFFFFFFF))
    s[15] = nonce_words[1]
    s[16] = nonce_words[2]

    r = @MArray zeros(UInt32, 16)
    @inbounds for i in 1:16
        r[i] = s[i]
    end

    @inbounds for _ in 1:10
        # QUARTERROUND(a,b,c,d)  (0-based CUDA indices, we +1 for Julia)
        # (0,4,8,12)
        r[1] += r[5]; r[13] = rotl32(r[13] ⊻ r[1], Int32(16))
        r[9] += r[13]; r[5] = rotl32(r[5] ⊻ r[9], Int32(12))
        r[1] += r[5]; r[13] = rotl32(r[13] ⊻ r[1], Int32(8))
        r[9] += r[13]; r[5] = rotl32(r[5] ⊻ r[9], Int32(7))
        # (1,5,9,13)
        r[2] += r[6]; r[14] = rotl32(r[14] ⊻ r[2], Int32(16))
        r[10] += r[14]; r[6] = rotl32(r[6] ⊻ r[10], Int32(12))
        r[2] += r[6]; r[14] = rotl32(r[14] ⊻ r[2], Int32(8))
        r[10] += r[14]; r[6] = rotl32(r[6] ⊻ r[10], Int32(7))
        # (2,6,10,14)
        r[3] += r[7]; r[15] = rotl32(r[15] ⊻ r[3], Int32(16))
        r[11] += r[15]; r[7] = rotl32(r[7] ⊻ r[11], Int32(12))
        r[3] += r[7]; r[15] = rotl32(r[15] ⊻ r[3], Int32(8))
        r[11] += r[15]; r[7] = rotl32(r[7] ⊻ r[11], Int32(7))
        # (3,7,11,15)
        r[4] += r[8]; r[16] = rotl32(r[16] ⊻ r[4], Int32(16))
        r[12] += r[16]; r[8] = rotl32(r[8] ⊻ r[12], Int32(12))
        r[4] += r[8]; r[16] = rotl32(r[16] ⊻ r[4], Int32(8))
        r[12] += r[16]; r[8] = rotl32(r[8] ⊻ r[12], Int32(7))
        # (0,5,10,15)
        r[1] += r[6]; r[16] = rotl32(r[16] ⊻ r[1], Int32(16))
        r[11] += r[16]; r[6] = rotl32(r[6] ⊻ r[11], Int32(12))
        r[1] += r[6]; r[16] = rotl32(r[16] ⊻ r[1], Int32(8))
        r[11] += r[16]; r[6] = rotl32(r[6] ⊻ r[11], Int32(7))
        # (1,6,11,12)
        r[2] += r[7]; r[13] = rotl32(r[13] ⊻ r[2], Int32(16))
        r[12] += r[13]; r[7] = rotl32(r[7] ⊻ r[12], Int32(12))
        r[2] += r[7]; r[13] = rotl32(r[13] ⊻ r[2], Int32(8))
        r[12] += r[13]; r[7] = rotl32(r[7] ⊻ r[12], Int32(7))
        # (2,7,8,13)
        r[3] += r[8]; r[14] = rotl32(r[14] ⊻ r[3], Int32(16))
        r[9] += r[14]; r[8] = rotl32(r[8] ⊻ r[9], Int32(12))
        r[3] += r[8]; r[14] = rotl32(r[14] ⊻ r[3], Int32(8))
        r[9] += r[14]; r[8] = rotl32(r[8] ⊻ r[9], Int32(7))
        # (3,4,9,14)
        r[4] += r[5]; r[15] = rotl32(r[15] ⊻ r[4], Int32(16))
        r[10] += r[15]; r[5] = rotl32(r[5] ⊻ r[10], Int32(12))
        r[4] += r[5]; r[15] = rotl32(r[15] ⊻ r[4], Int32(8))
        r[10] += r[15]; r[5] = rotl32(r[5] ⊻ r[10], Int32(7))
    end

    @inbounds for i in 1:16
        r[i] += s[i]
    end

    # unpack r[1..16] into out_bytes[block_start+1 .. +64] little-endian
    base = Int32(counter) * Int32(64)
    @inbounds for i in 1:16
        w = r[i]
        b0 = UInt8(w & UInt32(0xff))
        b1 = UInt8((w >> 8) & UInt32(0xff))
        b2 = UInt8((w >> 16) & UInt32(0xff))
        b3 = UInt8((w >> 24) & UInt32(0xff))
        # only write within bounds; caller passes num_bytes >= num_blocks*64 already
        idx = base + Int32((i - 1) * 4)
        out_bytes[idx + 1] = b0
        out_bytes[idx + 2] = b1
        out_bytes[idx + 3] = b2
        out_bytes[idx + 4] = b3
    end
    return
end

const H_KEY = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
const H_NONCE = "0001020304050607"
const H_KEYSTREAM = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9"

function hex_to_bytes(s::AbstractString)
    n = length(s) ÷ 2
    b = Vector{UInt8}(undef, n)
    for i in 1:n
        b[i] = parse(UInt8, s[2i-1:2i], base=16)
    end
    return b
end

function pack_key_words(key::Vector{UInt8})
    words = Vector{UInt32}(undef, 8)
    for i in 1:8
        base = (i-1)*4
        words[i] = UInt32(key[base+1]) |
                   (UInt32(key[base+2]) << 8) |
                   (UInt32(key[base+3]) << 16) |
                   (UInt32(key[base+4]) << 24)
    end
    return words
end

function pack_nonce_words(nonce::Vector{UInt8})
    words = Vector{UInt32}(undef, 2)
    for i in 1:2
        base = (i-1)*4
        words[i] = UInt32(nonce[base+1]) |
                   (UInt32(nonce[base+2]) << 8) |
                   (UInt32(nonce[base+3]) << 16) |
                   (UInt32(nonce[base+4]) << 24)
    end
    return words
end

function main()
    if length(ARGS) < 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat_n = parse(Int, ARGS[1])

    key_bytes = hex_to_bytes(H_KEY)
    nonce_bytes = hex_to_bytes(H_NONCE)
    ref_keystream = hex_to_bytes(H_KEYSTREAM)
    result_len = length(ref_keystream)
    num_blocks = cld(result_len, 64)
    total_bytes = num_blocks * 64  # allocate whole blocks

    key_words = pack_key_words(key_bytes)
    nonce_words = pack_nonce_words(nonce_bytes)

    d_key   = CuArray(key_words)
    d_nonce = CuArray(nonce_words)
    d_out   = CUDA.zeros(UInt8, total_bytes)

    threads = 256  # more than num_blocks (~6), extras exit
    # Warmup
    @cuda threads=threads blocks=1 chacha_kernel!(d_key, d_nonce, d_out, Int32(num_blocks))
    CUDA.synchronize()

    t0 = time_ns()
    for _ in 1:repeat_n
        CUDA.fill!(d_out, UInt8(0))
        @cuda threads=threads blocks=1 chacha_kernel!(d_key, d_nonce, d_out, Int32(num_blocks))
    end
    CUDA.synchronize()
    ktime_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average execution time of kernels: %f (us)\n", ktime_us)

    out = Array(d_out)
    ok = true
    for i in 1:result_len
        if out[i] != ref_keystream[i]
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
