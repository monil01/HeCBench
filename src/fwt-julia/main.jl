using CUDA
using Printf
using Random

function usage()
    println("Usage: ", PROGRAM_FILE, " <repeat> [log2Data log2Kernel]")
    exit(1)
end

function fwt_stage_kernel!(data, stride::Int32, pairs::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    step = blockDim().x * gridDim().x
    p = tid
    while p < pairs
        base = (p ÷ stride) * (stride << Int32(1))
        j = p - (p ÷ stride) * stride
        i0 = base + j + Int32(1)
        i1 = i0 + stride
        a = @inbounds data[i0]
        b = @inbounds data[i1]
        @inbounds data[i0] = a + b
        @inbounds data[i1] = a - b
        p += step
    end
    return
end

function modulate_kernel!(a, b, n::Int32, rcpn::Float32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    i = tid
    while i <= n
        @inbounds a[i] = a[i] * b[i] * rcpn
        i += step
    end
    return
end

function fwt_gpu!(d_data::CuArray{Float32}, log2n::Int)
    n = length(d_data)
    pairs = Int32(n ÷ 2)
    threads = 256
    blocks = min(4096, max(1, cld(Int(pairs), threads)))
    stride = Int32(n ÷ 2)
    while stride >= 1
        @cuda threads=threads blocks=blocks fwt_stage_kernel!(d_data, stride, pairs)
        stride ÷= Int32(2)
    end
    return d_data
end

function fwt_cpu!(data::Vector{Float32})
    n = length(data)
    stride = n ÷ 2
    while stride >= 1
        @inbounds for base in 1:(2 * stride):n
            for j in 0:(stride - 1)
                i0 = base + j
                i1 = i0 + stride
                a = data[i0]
                b = data[i1]
                data[i0] = a + b
                data[i1] = a - b
            end
        end
        stride ÷= 2
    end
    return data
end

function dyadic_convolution_fwt(data::Vector{Float32}, kernel::Vector{Float32}, log2data::Int)
    n = 1 << log2data
    a = copy(data)
    b = zeros(Float32, n)
    copyto!(b, 1, kernel, 1, length(kernel))
    fwt_cpu!(a)
    fwt_cpu!(b)
    @inbounds @simd for i in eachindex(a)
        a[i] = a[i] * b[i] / Float32(n)
    end
    fwt_cpu!(a)
    return a
end

function parse_args(args)
    length(args) in (1, 3) || usage()
    repeat = parse(Int, args[1])
    log2data = length(args) >= 2 ? parse(Int, args[2]) : 23
    log2kernel = length(args) >= 3 ? parse(Int, args[3]) : 7
    repeat > 0 || error("repeat must be positive")
    log2data >= log2kernel || error("log2Data must be >= log2Kernel")
    return repeat, log2data, log2kernel
end

function main()
    repeat, log2data, log2kernel = parse_args(ARGS)
    CUDA.allowscalar(false)
    datan = 1 << log2data
    kerneln = 1 << log2kernel
    @printf("Data length: %i; kernel length: %i\n", datan, kerneln)
    println("Initializing data...")
    rng = MersenneTwister(123)
    h_kernel = rand(rng, Float32, kerneln)
    h_data = rand(rng, Float32, datan)

    println("Running GPU dyadic convolution using Fast Walsh Transform...")
    d_data = CuArray(h_data)
    d_kernel = CUDA.zeros(Float32, datan)
    total_ns = 0
    for _ in 1:repeat
        copyto!(d_data, h_data)
        CUDA.fill!(d_kernel, 0.0f0)
        copyto!(view(d_kernel, 1:kerneln), h_kernel)
        CUDA.synchronize()
        start = time_ns()
        fwt_gpu!(d_data, log2data)
        fwt_gpu!(d_kernel, log2data)
        threads = 256
        blocks = min(4096, cld(datan, threads))
        @cuda threads=threads blocks=blocks modulate_kernel!(d_data, d_kernel, Int32(datan), Float32(1 / datan))
        fwt_gpu!(d_data, log2data)
        CUDA.synchronize()
        total_ns += time_ns() - start
    end
    @printf("Average device execution time %f (s)\n", total_ns * 1.0e-9 / repeat)

    println("Reading back GPU results...")
    h_result_gpu = Array(d_data)
    println("Running straightforward CPU dyadic convolution...")
    h_result_cpu = dyadic_convolution_fwt(h_data, h_kernel, log2data)
    println("Comparing the results...")
    sum_delta2 = 0.0
    sum_ref2 = 0.0
    @inbounds for i in eachindex(h_result_cpu)
        delta = Float64(h_result_cpu[i] - h_result_gpu[i])
        ref = Float64(h_result_cpu[i])
        sum_delta2 += delta * delta
        sum_ref2 += ref * ref
    end
    l2norm = sqrt(sum_delta2 / sum_ref2)
    println("Shutting down...")
    @printf("L2 norm: %E\n", l2norm)
    println(l2norm < 1.0e-5 ? "PASS" : "FAIL")
    l2norm < 1.0e-5 || exit(1)
end

main()
