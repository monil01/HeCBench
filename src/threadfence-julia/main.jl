using CUDA
using Printf

const BLOCKS = 256

function partial_sum_kernel!(array, result, n::Int32)
    shared = CuStaticSharedArray(Float32, BLOCKS)
    lid = threadIdx().x
    gid = (blockIdx().x - Int32(1)) * blockDim().x + lid

    value = Float32(0)
    if gid <= n
        value = array[gid]
    end
    shared[lid] = value
    sync_threads()

    stride = blockDim().x >>> Int32(1)
    while stride > 0
        if lid <= stride
            shared[lid] += shared[lid + stride]
        end
        sync_threads()
        stride >>>= Int32(1)
    end

    if lid == Int32(1)
        result[blockIdx().x] = shared[1]
    end
    return
end

function finalize_sum_kernel!(result, grids::Int32)
    shared = CuStaticSharedArray(Float32, BLOCKS)
    lid = threadIdx().x
    total = Float32(0)
    i = lid
    while i <= grids
        total += result[i]
        i += blockDim().x
    end
    shared[lid] = total
    sync_threads()

    stride = blockDim().x >>> Int32(1)
    while stride > 0
        if lid <= stride
            shared[lid] += shared[lid + stride]
        end
        sync_threads()
        stride >>>= Int32(1)
    end

    if lid == Int32(1)
        result[1] = shared[1]
    end
    return
end

function parse_args()
    if length(ARGS) != 2
        println("Usage: main.jl <repeat> <array length>")
        exit(1)
    end
    return parse(Int, ARGS[1]), parse(Int, ARGS[2])
end

function main()
    repeat, n = parse_args()
    grids = cld(n, BLOCKS)

    d_array = CUDA.fill(Float32(-1), n)
    d_result = CUDA.zeros(Float32, grids)

    ok = true
    elapsed_ns = 0.0
    CUDA.synchronize()

    for _ in 1:repeat
        start = time_ns()
        @cuda threads=BLOCKS blocks=grids partial_sum_kernel!(d_array, d_result, Int32(n))
        @cuda threads=BLOCKS blocks=1 finalize_sum_kernel!(d_result, Int32(grids))
        CUDA.synchronize()
        elapsed_ns += Float64(time_ns() - start)
    end

    h_sum = CUDA.@allowscalar d_result[1]
    ok = h_sum == Float32(-n)
    if ok
        @printf("Average kernel execution time: %f (ms)\n", (elapsed_ns * 1.0e-6) / repeat)
    end
    println(ok ? "PASS" : "FAIL")
end

main()
