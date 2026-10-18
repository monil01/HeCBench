using CUDA
using Printf

const MAX_GPU_COUNT = 8
const DEFAULT_DATA_N = 1048576 * 32

function reduce_kernel!(partial, data, n::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    thread_n = gridDim().x * blockDim().x
    acc = 0.0f0
    pos = tid0
    while pos < n
        @inbounds acc += data[pos + Int32(1)]
        pos += thread_n
    end
    @inbounds partial[tid0 + Int32(1)] = acc
    return
end

function libc_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])
    data_n = parse(Int, get(ENV, "SIMPLE_MULTI_DATA_N", string(DEFAULT_DATA_N)))
    block_n = 32
    thread_n = 256
    accum_n = block_n * thread_n

    println("Starting simpleMultiDevice")
    devices = collect(CUDA.devices())
    gpu_n = min(length(devices), MAX_GPU_COUNT)
    println("GPU device count: $gpu_n")
    println("Generating input data of size $data_n ...")
    println()

    data_counts = fill(data_n ÷ gpu_n, gpu_n)
    for i in 1:(data_n % gpu_n)
        data_counts[i] += 1
    end

    libc_srand(UInt32(1))
    host_data = Vector{Vector{Float32}}(undef, gpu_n)
    for i in 1:gpu_n
        host_data[i] = Vector{Float32}(undef, data_counts[i])
        for j in eachindex(host_data[i])
            host_data[i][j] = Float32(libc_rand()) / Float32(typemax(Cint))
        end
    end

    println("Computing with $gpu_n GPUs...")
    partials = Vector{Vector{Float32}}(undef, gpu_n)
    start = time_ns()
    for _ in 1:repeat
        for i in 1:gpu_n
            device!(devices[i])
            d_data = CuArray(host_data[i])
            d_sum = CUDA.zeros(Float32, accum_n)
            @cuda threads=thread_n blocks=block_n reduce_kernel!(d_sum, d_data, Int32(data_counts[i]))
            CUDA.synchronize()
            partials[i] = Array(d_sum)
        end
    end
    elapsed = time_ns() - start
    @printf("  Average GPU Processing time: %f (us)\n\n", elapsed * 1e-3 / repeat)

    h_sum_gpu = [sum(p) for p in partials]
    sum_gpu = sum(h_sum_gpu)

    println("Computing with Host CPU...")
    println()
    sum_cpu = sum(Float64(sum(v)) for v in host_data)
    println("Comparing GPU and Host CPU results...")
    diff = abs(sum_cpu - sum_gpu) / abs(sum_cpu)
    @printf("  GPU sum: %f\n  CPU sum: %f\n", sum_gpu, sum_cpu)
    @printf("  Relative difference: %E \n\n", diff)
    return diff < 1e-5 ? 0 : 1
end

exit(main())
