using CUDA
using Printf

const VECTOR_SIZE = 8 * 1024 * 1024
const GRANULARITY = 8
const FUSION_DEGREE = 4
const SEED = 0.1f0

function benchmark_kernel!(data, compute_iterations::Int32)
    block_size = blockDim().x
    stride = block_size
    idx0 = (blockIdx().x - Int32(1)) * block_size * Int32(GRANULARITY) + threadIdx().x
    big_stride = gridDim().x * block_size * Int32(GRANULARITY)

    for k in Int32(0):Int32(FUSION_DEGREE - 1)
        tmp1 = @inbounds data[idx0 + Int32(0) * stride + k * big_stride]
        tmp2 = @inbounds data[idx0 + Int32(1) * stride + k * big_stride]
        tmp3 = @inbounds data[idx0 + Int32(2) * stride + k * big_stride]
        tmp4 = @inbounds data[idx0 + Int32(3) * stride + k * big_stride]
        tmp5 = @inbounds data[idx0 + Int32(4) * stride + k * big_stride]
        tmp6 = @inbounds data[idx0 + Int32(5) * stride + k * big_stride]
        tmp7 = @inbounds data[idx0 + Int32(6) * stride + k * big_stride]
        tmp8 = @inbounds data[idx0 + Int32(7) * stride + k * big_stride]

        for _ in Int32(1):compute_iterations
            tmp1 = tmp1 * tmp1 + SEED
            tmp2 = tmp2 * tmp2 + SEED
            tmp3 = tmp3 * tmp3 + SEED
            tmp4 = tmp4 * tmp4 + SEED
            tmp5 = tmp5 * tmp5 + SEED
            tmp6 = tmp6 * tmp6 + SEED
            tmp7 = tmp7 * tmp7 + SEED
            tmp8 = tmp8 * tmp8 + SEED
        end

        s = tmp1 * tmp2 + tmp3 * tmp4 + tmp5 * tmp6 + tmp7 * tmp8
        @inbounds data[idx0 + k * big_stride] = s
    end
    return
end

function mixbench_gpu(size::Int, compute_iterations::Int, repeat::Int)
    println("Trade-off type:compute with global memory (block strided)")
    d_data = CUDA.zeros(Float32, size)
    reduced_grid_size = size ÷ GRANULARITY ÷ 128
    block_dim = 256
    grid_dim = reduced_grid_size ÷ block_dim

    for _ in 1:repeat
        @cuda threads=block_dim blocks=grid_dim benchmark_kernel!(d_data, Int32(compute_iterations))
    end

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_dim blocks=grid_dim benchmark_kernel!(d_data, Int32(compute_iterations))
    end
    CUDA.synchronize()
    total_s = (time_ns() - start) * 1.0e-9
    @printf("Total kernel execution time: %f (s)\n", total_s)

    data = Array(d_data)
    ok = true
    for (i, v) in pairs(data)
        if v != 0.0f0 && abs(v - 0.050807f0) > 1.0f-6
            @printf("Verification failed at index %d: %f\n", i - 1, v)
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <compute iterations> <repeat>")
        return 1
    end
    compute_iterations = parse(Int, args[1])
    repeat = parse(Int, args[2])
    datasize = VECTOR_SIZE * sizeof(Float32)
    @printf("Buffer size: %dMB\n", datasize ÷ (1024 * 1024))
    ok = mixbench_gpu(VECTOR_SIZE, compute_iterations, repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
