using CUDA
using Printf

function init_input!(src, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    while i < n
        src[i + Int32(1)] = i - (n ÷ Int32(2))
        i += stride
    end
    return
end

function filter_shared!(dst, nres, src, n::Int32)
    l_n = CUDA.@cuStaticSharedMem(Int32, 1)
    tx0 = threadIdx().x - Int32(1)
    i = (blockIdx().x - Int32(1)) * blockDim().x + tx0

    if tx0 == Int32(0)
        l_n[1] = Int32(0)
    end
    sync_threads()

    d = Int32(0)
    pos = Int32(0)
    keep = false
    if i < n
        d = src[i + Int32(1)]
        keep = d > Int32(0)
        if keep
            pos = CUDA.atomic_add!(pointer(l_n, 1), Int32(1))
        end
    end
    sync_threads()

    if tx0 == Int32(0)
        old = CUDA.atomic_add!(pointer(nres, 1), l_n[1])
        l_n[1] = old
    end
    sync_threads()

    if keep
        dst[l_n[1] + pos + Int32(1)] = d
    end
    return
end

function filter_global!(dst, nres, src, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < n
        d = src[i + Int32(1)]
        if d > Int32(0)
            pos = CUDA.atomic_add!(pointer(nres, 1), Int32(1))
            dst[pos + Int32(1)] = d
        end
    end
    return
end

function check_kernel!(dst, nres, sum_out, flag, expected_count::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    count = nres[1]
    if idx == Int32(1)
        if count != expected_count
            flag[1] = Int32(1)
        end
    end
    if idx <= count
        v = dst[idx]
        if v <= Int32(0)
            flag[1] = Int32(1)
        end
        CUDA.@atomic sum_out[1] += Int64(v)
    end
    return
end

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function verify_output(dst, nres, n::Int32)
    expected_count = n - (n ÷ Int32(2)) - Int32(1)
    hi = n - (n ÷ Int32(2)) - Int32(1)
    expected_sum = (Int64(hi) * Int64(hi + Int32(1))) ÷ 2
    sum_out = CuArray([Int64(0)])
    flag = CuArray([Int32(0)])
    blocks = max(ceil_div(Int(expected_count), 256), 1)
    @cuda threads=256 blocks=blocks check_kernel!(dst, nres, sum_out, flag, expected_count)
    CUDA.synchronize()
    return CUDA.@allowscalar(flag[1] == Int32(0) && sum_out[1] == expected_sum)
end

function run_filter!(kernel!, label::String, dst, nres, src, n::Int32, block_size::Int, repeat::Int)
    blocks = ceil_div(Int(n), block_size)
    fill!(nres, Int32(0))
    @cuda threads=block_size blocks=blocks kernel!(dst, nres, src, n)
    CUDA.synchronize()

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        fill!(nres, Int32(0))
        @cuda threads=block_size blocks=blocks kernel!(dst, nres, src, n)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6
    @printf("Average execution time of %s %lf (ms)\n", label, elapsed_ms / repeat)
    ok = verify_output(dst, nres, n)
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of elements> <block size> <repeat>")
        return 1
    end

    n = parse(Int32, args[1])
    block_size = parse(Int, args[2])
    repeat = parse(Int, args[3])

    src = CuArray{Int32}(undef, Int(n))
    dst = CuArray{Int32}(undef, Int(n))
    nres = CuArray([Int32(0)])

    blocks = ceil_div(Int(n), block_size)
    @cuda threads=block_size blocks=blocks init_input!(src, n)
    CUDA.synchronize()

    ok1 = run_filter!(filter_shared!, "filter (shared memory)", dst, nres, src, n, block_size, repeat)
    ok2 = run_filter!(filter_global!, "filter (global aggregate)", dst, nres, src, n, block_size, repeat)
    return (ok1 && ok2) ? 0 : 1
end

exit(main(ARGS))
