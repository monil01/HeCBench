using CUDA
using Printf

const NUM_BLOCKS = 1024
const BLOCK_SIZE = 256
const BOUND = NUM_BLOCKS * BLOCK_SIZE

function set_min_kernel!(res, value)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds res[1] = value
    end
    return
end

function set_max_kernel!(res, value)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds res[1] = value
    end
    return
end

function set_add_kernel!(res, value)
    if blockIdx().x == 1 && threadIdx().x == 1
        @inbounds res[1] = value
    end
    return
end

function timed_set!(kernel, res, value, repeat)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=NUM_BLOCKS kernel(res, value)
    end
    CUDA.synchronize()
    (time_ns() - t0) * 1.0e-9 / repeat
end

function main(args)
    if length(args) != 1
        println("Usage: ./main <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])
    sum_bound = UInt64(BOUND) * UInt64(BOUND + 1) ÷ UInt64(2)

    u64 = CuArray([typemax(UInt64), UInt64(0), UInt64(0)])
    s64 = CuArray([typemax(Int64), typemin(Int64), Int64(0)])
    f64 = CuArray([floatmax(Float64), floatmin(Float64), 0.0])

    t = timed_set!(set_min_kernel!, view(u64, 1:1), UInt64(1), repeat)
    @printf("Atomic min for data type U64 | Average execution time: %f (s)\n", t)
    t = timed_set!(set_min_kernel!, view(s64, 1:1), Int64(1), repeat)
    @printf("Atomic min for data type S64 | Average execution time: %f (s)\n", t)
    t = timed_set!(set_min_kernel!, view(f64, 1:1), 1.0, repeat)
    @printf("Atomic min for data type F64 | Average execution time: %f (s)\n", t)

    t = timed_set!(set_max_kernel!, view(u64, 2:2), UInt64(BOUND), repeat)
    @printf("Atomic max for data type U64 | Average execution time: %f (s)\n", t)
    t = timed_set!(set_max_kernel!, view(s64, 2:2), Int64(BOUND), repeat)
    @printf("Atomic max for data type S64 | Average execution time: %f (s)\n", t)
    t = timed_set!(set_max_kernel!, view(f64, 2:2), Float64(BOUND), repeat)
    @printf("Atomic max for data type F64 | Average execution time: %f (s)\n", t)

    t = timed_set!(set_add_kernel!, view(u64, 3:3), sum_bound, 1)
    @printf("Atomic add for data type U64 | Average execution time: %f (s)\n", t)
    t = timed_set!(set_add_kernel!, view(s64, 3:3), Int64(sum_bound), 1)
    @printf("Atomic add for data type S64 | Average execution time: %f (s)\n", t)
    t = timed_set!(set_add_kernel!, view(f64, 3:3), Float64(sum_bound), 1)
    @printf("Atomic add for data type F64 | Average execution time: %f (s)\n", t)

    hu = Array(u64)
    hs = Array(s64)
    hf = Array(f64)
    ok = hu[1] == UInt64(1) && hs[1] == Int64(1) && hf[1] == 1.0
    ok &= hu[2] == UInt64(BOUND) && hs[2] == Int64(BOUND) && hf[2] == Float64(BOUND)
    ok &= hu[3] == sum_bound && hs[3] == Int64(sum_bound) && hf[3] == Float64(sum_bound)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
