using CUDA
using Printf

const TREE_NUM = 4096
const TREE_SIZE = 4096
const GROUP_SIZE = 256

@inline function wrap_i32(x::Int64)
    return reinterpret(Int32, UInt32(x % (Int64(1) << 32)))
end

function aos_kernel!(out)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if gid < Int32(TREE_NUM)
        res = Int64(0)
        base = Int64(gid) * TREE_SIZE
        for j in 0:(TREE_SIZE - 1)
            res += base + j
        end
        @inbounds out[gid + Int32(1)] = wrap_i32(res)
    end
    return
end

function soa_kernel!(out)
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if gid < Int32(TREE_NUM)
        res = Int64(0)
        base = Int64(gid) * TREE_SIZE
        for j in 0:(TREE_SIZE - 1)
            res += base + j
        end
        @inbounds out[gid + Int32(1)] = wrap_i32(res)
    end
    return
end

function reference()
    ref = Vector{Int32}(undef, TREE_NUM)
    for i in 0:(TREE_NUM - 1)
        res = Int64(0)
        base = Int64(i) * TREE_SIZE
        for j in 0:(TREE_SIZE - 1)
            res += base + j
        end
        ref[i + 1] = wrap_i32(res)
    end
    ref
end

function run_kernel!(kernel, out, iterations)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=GROUP_SIZE blocks=(TREE_NUM ÷ GROUP_SIZE) kernel(out)
    end
    CUDA.synchronize()
    (time_ns() - t0) * 1.0e-3 / iterations
end

function main(args)
    if length(args) != 1
        println("Usage: ./main <repeat>")
        return 1
    end
    iterations = parse(Int, args[1])
    if iterations < 1
        println("Iterations cannot be 0 or negative. Exiting..")
        return -1
    end

    ref = reference()
    out = CUDA.zeros(Int32, TREE_NUM)

    t = run_kernel!(aos_kernel!, out, iterations)
    @printf("Average kernel execution time (AoS): %f (us)\n", t)
    println(Array(out) == ref ? "PASS" : "FAIL")

    t = run_kernel!(soa_kernel!, out, iterations)
    @printf("Average kernel execution time (SoA): %f (us)\n", t)
    println(Array(out) == ref ? "PASS" : "FAIL")
    return 0
end

exit(main(ARGS))
