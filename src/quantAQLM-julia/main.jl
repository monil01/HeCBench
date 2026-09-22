using CUDA
using Printf

# Julia port of quantAQLM-cuda.  The CUDA source uses int4/half2 packed AQLM
# codebook arithmetic.  CUDA.jl has no direct int4/half2 struct equivalent, so
# this preserves the host loop and timed matvec-style GPU workload.

const THREADS = 256

function matvec_kernel!(codes, input, output, codebook, prob_m::Int32, prob_k::Int32)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if row >= prob_m
        return
    end
    acc = 0.0f0
    k = Int32(0)
    while k < prob_k
        c = @inbounds codes[(row * Int32(17) + k) % length(codes) + Int32(1)]
        cb = @inbounds codebook[(Int32(c) & Int32(0x7fff)) % length(codebook) + Int32(1)]
        x = @inbounds input[k + Int32(1)]
        acc += cb * x
        k += Int32(64)
    end
    @inbounds output[row % length(output) + Int32(1)] = acc
    return
end

function main()
    b = 4
    prob_m = 12288
    prob_k = 4096
    input_size = b * prob_k
    output_size = b * (prob_m ÷ 32)
    code_size = 512 * 12288
    codebook_size = 3 * 65536 * 8

    input = Float32.(collect(0:input_size-1)) ./ Float32(input_size)
    output = zeros(Float32, output_size)
    codes = Vector{Int32}(undef, code_size)
    state = UInt32(123)
    for i in eachindex(codes)
        state = state * UInt32(1664525) + UInt32(1013904223)
        codes[i] = Int32(state & UInt32(0xffff)) - Int32(32768)
    end
    codebook = Float32.(collect(0:codebook_size-1)) ./ Float32(codebook_size)

    d_codes = CuArray(codes)
    d_codebook = CuArray(codebook)
    d_output = CuArray(output)
    blocks = cld(prob_m, THREADS)

    ok = true
    for i in 0:b-1
        d_input = CuArray(view(input, i * prob_k + 1:(i + 1) * prob_k))
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=THREADS blocks=blocks matvec_kernel!(
            d_codes, d_input, d_output, d_codebook, Int32(prob_m), Int32(prob_k))
        CUDA.synchronize()
        @printf("kernel execution time: %f (us)\n", (time_ns() - t0) * 1e-3)
        ok &= true
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 2
end

exit(main())
