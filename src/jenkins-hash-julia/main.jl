using CUDA
using Printf

rot(x::UInt32, k::Int) = (x << k) | (x >> (32 - k))

function mix_vals(a::UInt32, b::UInt32, c::UInt32)
    a -= c; a ⊻= rot(c, 4);  c += b
    b -= a; b ⊻= rot(a, 6);  a += c
    c -= b; c ⊻= rot(b, 8);  b += a
    a -= c; a ⊻= rot(c, 16); c += b
    b -= a; b ⊻= rot(a, 19); a += c
    c -= b; c ⊻= rot(b, 4);  b += a
    return a, b, c
end

function final_vals(a::UInt32, b::UInt32, c::UInt32)
    c ⊻= b; c -= rot(b, 14)
    a ⊻= c; a -= rot(c, 11)
    b ⊻= a; b -= rot(a, 25)
    c ⊻= b; c -= rot(b, 16)
    a ⊻= c; a -= rot(c, 4)
    b ⊻= a; b -= rot(a, 14)
    c ⊻= b; c -= rot(b, 24)
    return a, b, c
end

function mix_remainder(a::UInt32, b::UInt32, c::UInt32,
                       k0::UInt32, k1::UInt32, k2::UInt32, length::UInt32)
    if length == 12
        c += k2; b += k1; a += k0
    elseif length == 11
        c += k2 & 0x00ffffff; b += k1; a += k0
    elseif length == 10
        c += k2 & 0x0000ffff; b += k1; a += k0
    elseif length == 9
        c += k2 & 0x000000ff; b += k1; a += k0
    elseif length == 8
        b += k1; a += k0
    elseif length == 7
        b += k1 & 0x00ffffff; a += k0
    elseif length == 6
        b += k1 & 0x0000ffff; a += k0
    elseif length == 5
        b += k1 & 0x000000ff; a += k0
    elseif length == 4
        a += k0
    elseif length == 3
        a += k0 & 0x00ffffff
    elseif length == 2
        a += k0 & 0x0000ffff
    elseif length == 1
        a += k0 & 0x000000ff
    elseif length == 0
        return c
    end
    _, _, c = final_vals(a, b, c)
    return c
end

function hashlittle_words(keys::AbstractVector{UInt32}, offset::Int, length0::UInt32, initval::UInt32)
    length = length0
    a = UInt32(0xdeadbeef) + length + initval
    b = a
    c = a
    pos = offset
    while length > 12
        a += keys[pos]
        b += keys[pos + 1]
        c += keys[pos + 2]
        a, b, c = mix_vals(a, b, c)
        length -= UInt32(12)
        pos += 3
    end
    return mix_remainder(a, b, c, keys[pos], keys[pos + 1], keys[pos + 2], length)
end

function jenkins_kernel!(lengths, initvals, keys, out, n::Int64)
    id = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    if id <= n
        idx = Int(id)
        @inbounds out[idx] = hashlittle_words(keys, (idx - 1) * 16 + 1, lengths[idx], initvals[idx])
    end
    return
end

function libc_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function make_key_words()
    bytes = zeros(UInt8, 64)
    str = codeunits("Four score and seven years ago")
    for i in eachindex(str)
        bytes[i] = str[i]
    end
    words = Vector{UInt32}(undef, 16)
    for i in 0:15
        base = 4i + 1
        words[i + 1] = UInt32(bytes[base]) |
                       (UInt32(bytes[base + 1]) << 8) |
                       (UInt32(bytes[base + 2]) << 16) |
                       (UInt32(bytes[base + 3]) << 24)
    end
    return words
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <block size> <number of strings> <repeat>")
        return 1
    end

    block_size = parse(Int, ARGS[1])
    n = parse(Int64, ARGS[2])
    repeat = parse(Int, ARGS[3])

    key_words = make_key_words()
    sample_hash = hashlittle_words(key_words, 1, UInt32(30), UInt32(1))
    @printf("input string: %s hash is %.8x\n", "Four score and seven years ago", sample_hash)

    keys = Vector{UInt32}(undef, n * 16)
    lengths = Vector{UInt32}(undef, n)
    initvals = Vector{UInt32}(undef, n)

    libc_srand(UInt32(2))
    for i in 1:n
        copyto!(keys, (i - 1) * 16 + 1, key_words, 1, 16)
        lengths[i] = UInt32(mod(libc_rand(), 61))
        initvals[i] = UInt32((i - 1) % 2)
    end

    d_keys = CuArray(keys)
    d_lengths = CuArray(lengths)
    d_initvals = CuArray(initvals)
    d_out = CUDA.zeros(UInt32, n)
    grids = cld(n, block_size)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=grids jenkins_kernel!(
            d_lengths, d_initvals, d_keys, d_out, n)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time : %f (s)\n", elapsed * 1e-9 / repeat)

    out = Array(d_out)
    println("Verify the results computed on the device..")
    error = false
    for i in 1:n
        c = hashlittle_words(keys, (i - 1) * 16 + 1, lengths[i], initvals[i])
        if out[i] != c
            @printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i - 1, out[i], c)
            error = true
            break
        end
    end
    println(error ? "FAIL" : "PASS")
    return 0
end

exit(main())
