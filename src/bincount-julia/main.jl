using CUDA
using Printf
using Random

const THREADS_PER_BLOCK = 256

@inline function get_bin(v::Float32, minvalue::Float32, maxvalue::Float32, nbins::Int32)
    bin = Int32(trunc((v - minvalue) * Float32(nbins) / (maxvalue - minvalue)))
    if bin == nbins
        bin -= Int32(1)
    end
    return bin
end

function bincount_global_kernel!(output, input, nbins::Int32,
                                 minvalue::Float32, maxvalue::Float32,
                                 input_size::Int32)
    i0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = gridDim().x * blockDim().x
    while i0 < input_size
        @inbounds v = input[i0 + Int32(1)]
        if v >= minvalue && v <= maxvalue
            bin = get_bin(v, minvalue, maxvalue, nbins)
            CUDA.atomic_add!(pointer(output, bin + Int32(1)), Int32(1))
        end
        i0 += stride
    end
    return
end

function bincount_shared_kernel!(output, input, nbins::Int32,
                                 minvalue::Float32, maxvalue::Float32,
                                 input_size::Int32)
    smem = CuDynamicSharedArray(Int32, nbins)
    tid0 = threadIdx().x - Int32(1)
    j = tid0
    while j < nbins
        @inbounds smem[j + Int32(1)] = Int32(0)
        j += blockDim().x
    end
    sync_threads()

    i0 = (blockIdx().x - Int32(1)) * blockDim().x + tid0
    stride = gridDim().x * blockDim().x
    while i0 < input_size
        @inbounds v = input[i0 + Int32(1)]
        if v >= minvalue && v <= maxvalue
            bin = get_bin(v, minvalue, maxvalue, nbins)
            CUDA.atomic_add!(pointer(smem, bin + Int32(1)), Int32(1))
        end
        i0 += stride
    end
    sync_threads()

    j = tid0
    while j < nbins
        @inbounds CUDA.atomic_add!(pointer(output, j + Int32(1)), smem[j + Int32(1)])
        j += blockDim().x
    end
    return
end

function reference!(output, input, nbins::Int, minvalue::Float32, maxvalue::Float32, repeat::Int)
    for _ in 1:repeat
        for v in input
            if v >= minvalue && v <= maxvalue
                bin = Int(trunc((v - minvalue) * Float32(nbins) / (maxvalue - minvalue)))
                if bin == nbins
                    bin -= 1
                end
                output[bin + 1] += 1
            end
        end
    end
end

function run_eval(input_size::Int, repeat::Int)
    rng = MersenneTwister(123)
    input = randn(rng, Float32, input_size) .* 2.0f0 .+ 5.0f0
    input_min = minimum(input)
    input_max = maximum(input)
    @printf("Input min, max values: (%f %f)\n", input_min, input_max)

    d_input = CuArray(input)
    max_shared = 48 * 1024
    @printf("Maximum shared local memory size per block in bytes: %d\n", max_shared)

    blocks = cld(input_size, THREADS_PER_BLOCK)
    for nbins in (768, 1536, 3072, 6144, 12288, 24576)
        @printf("\nNumber of bins: %d\n", nbins)
        output_ref = zeros(Int32, nbins)
        reference!(output_ref, input, nbins, input_min, input_max, repeat)
        d_output = CUDA.zeros(Int32, nbins)

        println("bincount using global atomics")
        CUDA.synchronize()
        t0 = time_ns()
        for _ in 1:repeat
            @cuda threads=THREADS_PER_BLOCK blocks=blocks bincount_global_kernel!(
                d_output, d_input, Int32(nbins), input_min, input_max, Int32(input_size))
        end
        CUDA.synchronize()
        @printf("Average execution time of bincount kernel: %f (us)\n",
                (time_ns() - t0) * 1.0e-3 / repeat)
        output = Array(d_output)
        println(output == output_ref ? "PASS" : "FAIL")

        shared_mem = nbins * sizeof(Int32)
        if shared_mem <= max_shared
            println()
            println("bincount using global and local atomics")
            fill!(d_output, 0)
            CUDA.synchronize()
            t0 = time_ns()
            for _ in 1:repeat
                @cuda threads=THREADS_PER_BLOCK blocks=blocks shmem=shared_mem bincount_shared_kernel!(
                    d_output, d_input, Int32(nbins), input_min, input_max, Int32(input_size))
            end
            CUDA.synchronize()
            @printf("Average execution time of bincount kernel: %f (us)\n",
                    (time_ns() - t0) * 1.0e-3 / repeat)
            output = Array(d_output)
            println(output == output_ref ? "PASS" : "FAIL")
        end
    end
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end
    run_eval(parse(Int, args[1]), parse(Int, args[2]))
    return 0
end

exit(main(ARGS))
