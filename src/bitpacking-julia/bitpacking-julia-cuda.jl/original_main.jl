using CUDA
using Printf

const BLOCK_SIZE = 256
const NUM_PACK_REPEATS = 1000

function pack_kernel!(out, input, min_value::Int32, num_bits::UInt8, n::Int64, nwords::Int64)
    idx = Int64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    stride = Int64(gridDim().x * blockDim().x)
    bits_per_word = Int64(32)
    nb = Int64(num_bits)
    while idx < nwords
        bit_start = idx * bits_per_word
        bit_end = bit_start + bits_per_word
        start_idx = bit_start ÷ nb
        end_idx = cld(bit_end, nb)
        v = UInt32(0)
        @inbounds for item in start_idx:(end_idx - 1)
            if item < n
                raw = UInt32(input[Int(item + 1)] - min_value)
                off = item * nb - bit_start
                if off >= 0
                    v |= raw << UInt32(off)
                else
                    v |= raw >> UInt32(-off)
                end
            end
        end
        @inbounds out[Int(idx + 1)] = v
        idx += stride
    end
    return
end

function unpack_bytes(bytes::Vector{UInt8}, num_bits::UInt8, min_value::Int32, idx0::Int64)
    if num_bits == 0
        return min_value
    end
    nb = Int(num_bits)
    mask = UInt32(nb < 32 ? (UInt64(1) << nb) - UInt64(1) : typemax(UInt32))
    start_byte = (idx0 * nb) ÷ 8
    end_byte = ((idx0 + 1) * nb - 1) ÷ 8
    bit_offset = (idx0 * nb) % 8
    base = UInt32(0)
    @inbounds for (k, j) in enumerate(start_byte:end_byte)
        shifted = UInt32(bytes[j + 1])
        shift = (k - 1) * 8 - bit_offset
        if shift > 0
            shifted <<= UInt32(shift)
        else
            shifted >>= UInt32(-shift)
        end
        base |= mask & shifted
    end
    return Int32(base) + min_value
end

function bits_required(min_value::Int32, max_value::Int32)
    range = UInt32(max_value - min_value)
    return UInt8(range == 0 ? 0 : 32 - leading_zeros(range))
end

function fill_source!(source::Vector{Int32})
    x = UInt32(0)
    @inbounds for i in eachindex(source)
        x = UInt32(1664525) * x + UInt32(1013904223)
        source[i] = Int32(x & UInt32(0x7fffffff))
    end
    return source
end

function run_bitpacking(input_host::Vector{Int32}, num_bits_max::Int, n::Int)
    min_value = minimum(@view input_host[1:n])
    max_value = maximum(@view input_host[1:n])
    num_bits = bits_required(min_value, max_value)
    num_bits <= num_bits_max || error("computed bit width exceeds maximum")

    packed_size = (((num_bits_max * n) ÷ 64) + 1) * 8
    nwords = cld(packed_size, sizeof(UInt32))

    d_input = CuArray(@view input_host[1:n])
    d_output = CUDA.zeros(UInt32, nwords)
    threads = BLOCK_SIZE
    blocks = min(4096, cld(nwords, threads))

    @cuda threads=threads blocks=blocks pack_kernel!(d_output, d_input, min_value, num_bits, Int64(n), Int64(nwords))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:NUM_PACK_REPEATS
        @cuda threads=threads blocks=blocks pack_kernel!(d_output, d_input, min_value, num_bits, Int64(n), Int64(nwords))
    end
    CUDA.synchronize()
    @printf("Total kernel execution time (1000 iterations) = %f (s)\n", (time_ns() - t0) * 1e-9)

    output_words = Array(d_output)
    output_bytes = collect(reinterpret(UInt8, output_words))
    return output_bytes, Int(num_bits), min_value
end

function main()
    offset = Int32(87231)
    num_bits = 13
    sizes = (2, 123, 3411, 83621, 872163, 100000001)
    max_n = maximum(sizes)
    source = Vector{Int32}(undef, max_n)
    fill_source!(source)

    input_host = Vector{Int32}(undef, max_n)
    mask = Int32((UInt32(1) << UInt32(num_bits)) - UInt32(1))

    for n in sizes
        @inbounds for i in 1:n
            input_host[i] = (source[i] & mask) + offset
        end

        @printf("Size = %10d\n", n)
        offload_start = time_ns()
        output_bytes, num_bits_act, min_value = run_bitpacking(input_host, num_bits, n)
        @printf("Device offload time = %f (s)\n", (time_ns() - offload_start) * 1e-9)

        ok = num_bits_act <= num_bits
        num_samples = floor(Int, sqrt(n)) + 1
        @inbounds for i in 1:num_samples
            idx0 = Int(UInt32(source[i])) % n
            if unpack_bytes(output_bytes, UInt8(num_bits_act), min_value, Int64(idx0)) != input_host[idx0 + 1]
                ok = false
                break
            end
        end
        println(ok ? "PASS" : "FAIL")
    end
end

main()
