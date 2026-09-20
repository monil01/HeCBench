using CUDA
using Printf

const BLOCK_SIZE = Int32(256)
const BASE_NUMEL = Int64(1024) * 1024 * 256 + 2

function init_kernel!(data, n::Int64)
    idx = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x)
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx
    while i <= n
        data[i] = convert(eltype(data), i % Int64(101))
        i += stride
    end
    return
end

function thread_copy_kernel!(input, output, n::Int64, vec_size::Int32)
    tid = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    index = tid * Int64(vec_size) + 1
    j = Int32(0)
    while j < vec_size
        pos = index + Int64(j)
        if pos <= n
            output[pos] = input[pos]
        end
        j += Int32(1)
    end
    return
end

function compare_kernel!(input, output, flag, n::Int64)
    idx = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x)
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx
    while i <= n
        if input[i] != output[i]
            flag[1] = Int32(1)
        end
        i += stride
    end
    return
end

function ceil_div(a::Integer, b::Integer)
    return (a + b - 1) ÷ b
end

function test_threads_copy(::Type{T}, vec_size::Int32, n::Int64, repeat::Int) where {T}
    input = CuArray{T}(undef, n)
    output = CuArray{T}(undef, n)

    init_blocks = min(ceil_div(n, Int64(BLOCK_SIZE)), Int64(65535))
    @cuda threads=Int(BLOCK_SIZE) blocks=Int(init_blocks) init_kernel!(input, n)
    CUDA.synchronize()

    block_work_size = Int64(BLOCK_SIZE) * Int64(vec_size)
    blocks = ceil_div(n, block_work_size)

    for _ in 1:100
        @cuda threads=Int(BLOCK_SIZE) blocks=Int(blocks) thread_copy_kernel!(input, output, n, vec_size)
    end
    CUDA.synchronize()

    flag = CuArray([Int32(0)])
    @cuda threads=Int(BLOCK_SIZE) blocks=Int(init_blocks) compare_kernel!(input, output, flag, n)
    CUDA.synchronize()
    passed = CUDA.@allowscalar flag[1] == 0
    println(passed ? "PASS" : "FAIL")

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=Int(BLOCK_SIZE) blocks=Int(blocks) thread_copy_kernel!(input, output, n, vec_size)
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - start
    avg_time_ms = (elapsed_ns * 1.0e-6) / repeat
    total_gbytes = 2 * n * sizeof(T) / 1000.0 / 1000.0
    @printf("Average kernel execution time (ms):%f Throughput:%f GB/s\n",
            avg_time_ms, total_gbytes / avg_time_ms)

    return passed
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])

    println("1GB threads copy test ...")
    all_passed = true

    for vec_size in Int32[1, 2, 4, 8, 16]
        print("int$(vec_size): ")
        all_passed &= test_threads_copy(Int32, vec_size, BASE_NUMEL, repeat)
    end

    for vec_size in Int32[1, 2, 4, 8, 16]
        print("short$(vec_size): ")
        all_passed &= test_threads_copy(Int16, vec_size, BASE_NUMEL * 2, repeat)
    end

    for vec_size in Int32[1, 2, 4, 8, 16]
        print("char$(vec_size): ")
        all_passed &= test_threads_copy(Int8, vec_size, BASE_NUMEL * 4, repeat)
    end

    return all_passed ? 0 : 1
end

exit(main(ARGS))
