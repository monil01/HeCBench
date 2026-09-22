using CUDA
using Printf

const THREADS_PER_BLOCK = 256
const LAUNCH_ITERATIONS = 3

function reduce_kernel!(input, partial, input_size::Int32)
    bid = blockIdx().x
    tid = threadIdx().x
    global_tid = Int32((bid - 1) * blockDim().x + tid)
    stride = Int32(blockDim().x * gridDim().x)
    acc = 0.0
    i = global_tid
    while i <= input_size
        @inbounds acc += Float64(input[i])
        i += stride
    end
    sh = @cuStaticSharedMem(Float64, THREADS_PER_BLOCK)
    sh[tid] = acc
    sync_threads()
    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            sh[tid] += sh[tid + offset]
        end
        sync_threads()
        offset ÷= 2
    end
    if tid == 1
        @inbounds partial[bid] = sh[1]
    end
    return
end

function reduce_final_kernel!(partial, result, nblocks::Int32)
    tid = threadIdx().x
    acc = 0.0
    i = tid
    while i <= nblocks
        @inbounds acc += partial[i]
        i += blockDim().x
    end
    sh = @cuStaticSharedMem(Float64, THREADS_PER_BLOCK)
    sh[tid] = acc
    sync_threads()
    offset = blockDim().x ÷ 2
    while offset > 0
        if tid <= offset
            sh[tid] += sh[tid + offset]
        end
        sync_threads()
        offset ÷= 2
    end
    if tid == 1
        result[1] = sh[1]
    end
    return
end

function init_input(size::Int)
    state = UInt32(123)
    out = Vector{Float32}(undef, size)
    @inbounds for i in 1:size
        state = state * UInt32(1103515245) + UInt32(12345)
        out[i] = Float32((state >> 16) & UInt32(0xff)) / Float32(2147483647)
    end
    return out
end

function run_case(label::String, input_h, max_blocks::Int, inner_iters::Int)
    result_ref = sum(Float64, input_h)
    input_d = CuArray(input_h)
    partial = CUDA.zeros(Float64, max_blocks)
    result = CUDA.zeros(Float64, 1)
    for _ in 1:LAUNCH_ITERATIONS
        CUDA.synchronize()
        t0 = time_ns()
        for _ in 1:inner_iters
            fill!(partial, 0.0)
            @cuda threads=THREADS_PER_BLOCK blocks=max_blocks reduce_kernel!(input_d, partial, Int32(length(input_h)))
            @cuda threads=THREADS_PER_BLOCK blocks=1 reduce_final_kernel!(partial, result, Int32(max_blocks))
        end
        CUDA.synchronize()
        elapsed = (time_ns() - t0) * 1e-3
        result_h = Array(result)[1]
        println(abs(result_h - result_ref) < 1.0e-4 ? "PASS" : "FAIL")
        @printf("Execution time of using %s: %f (us)\n\n", label, elapsed)
    end
end

function main(args)
    max_size = length(args) >= 1 ? parse(Int, args[1]) : 1 << 27
    inner_iters = length(args) >= 2 ? parse(Int, args[2]) : 100
    max_blocks = 512
    size = 512
    while size <= max_size
        println()
        println("-----------------------------")
        println("$size elements")
        println("Threads per block  = $THREADS_PER_BLOCK")
        println("Launch iterations = $LAUNCH_ITERATIONS")
        input = init_input(size)
        run_case("Graph", input, max_blocks, inner_iters)
        run_case("Stream", input, max_blocks, inner_iters)
        size *= 512
    end
    return 0
end

exit(main(ARGS))
