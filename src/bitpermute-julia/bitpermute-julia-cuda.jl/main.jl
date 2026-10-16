using CUDA
using Printf

const MAX_LG_DOMAIN_SIZE = 28
const THREADS = 256

function bit_reverse32(x::UInt32, nbits::UInt32)
    y = UInt32(0)
    for _ in UInt32(0):UInt32(31)
        y = (y << UInt32(1)) | (x & UInt32(1))
        x >>= UInt32(1)
    end
    return y >> (UInt32(32) - nbits)
end

function bit_rev_permutation!(out, inp, lg_domain_size::UInt32, domain_size::UInt32)
    idx0 = UInt32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = UInt32(blockDim().x * gridDim().x)

    while idx0 < domain_size
        rev = bit_reverse32(idx0, lg_domain_size)
        if idx0 < rev
            a = inp[Int(idx0) + 1]
            b = inp[Int(rev) + 1]
            out[Int(idx0) + 1] = b
            out[Int(rev) + 1] = a
        elseif idx0 == rev
            out[Int(idx0) + 1] = inp[Int(idx0) + 1]
        end
        idx0 += stride
    end
    return
end

function bit_reverse_host(x::UInt32, nbits::UInt32)
    y = UInt32(0)
    for _ in 0:31
        y = (y << 1) | (x & 0x00000001)
        x >>= 1
    end
    return y >> (UInt32(32) - nbits)
end

function bit_rev_cpu(inp::Vector{Int64}, lg_domain_size::Int)
    domain_size = 1 << lg_domain_size
    out = copy(inp)
    for i0 in UInt32(0):UInt32(domain_size - 1)
        r = bit_reverse_host(i0, UInt32(lg_domain_size))
        if i0 < r
            out[Int(r) + 1] = inp[Int(i0) + 1]
            out[Int(i0) + 1] = inp[Int(r) + 1]
        end
    end
    return out
end

function bit_rev!(d_inout, lg_domain_size::Int)
    domain_size = UInt32(1) << UInt32(lg_domain_size)
    blocks = min(cld(Int(domain_size), THREADS), 65_535)
    @cuda threads=THREADS blocks=blocks bit_rev_permutation!(
        d_inout, d_inout, UInt32(lg_domain_size), domain_size)
end

function bit_permute(lg_domain_size::Int, repeat::Int)
    if lg_domain_size > MAX_LG_DOMAIN_SIZE
        error("lg_domain_size exceeds MAX_LG_DOMAIN_SIZE")
    end

    domain_size = 1 << lg_domain_size
    println("Domain size is $domain_size")

    h_inout = Int64.(0:(domain_size - 1))
    h_ref = bit_rev_cpu(h_inout, lg_domain_size)
    d_inout = CuArray(h_inout)

    bit_rev!(d_inout, lg_domain_size)
    CUDA.synchronize()
    h_check = Array(d_inout)
    ok = h_check == h_ref
    println(ok ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        bit_rev!(d_inout, lg_domain_size)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of the kernel: %f (us)\n\n", elapsed_us)
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    bit_permute(10, repeat)
    bit_permute(11, repeat)
    bit_permute(15, repeat)
    bit_permute(27, repeat)
    bit_permute(28, repeat)
end

main()
