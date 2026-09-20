using CUDA
using Printf

function input_value(idx0::Int64)
    return Float32(idx0 % Int64(1024)) / 1024.0f0
end

function init_input!(input, total::Int64)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        input[i + 1] = input_value(i)
        i += stride
    end
    return
end

function welford_kernel!(input, mean, var, bs::Int32, fs::Int32, ss::Int32)
    fid = blockIdx().x - Int32(1)
    tx = threadIdx().x - Int32(1)
    local_sum = 0.0f0
    local_sq = 0.0f0
    total = bs * ss
    i = tx
    while i < total
        bid = i ÷ ss
        s = i - bid * ss
        idx = Int64(bid) * Int64(ss) * Int64(fs) + Int64(fid) * Int64(ss) + Int64(s)
        v = input[idx + 1]
        local_sum += v
        local_sq += v * v
        i += blockDim().x
    end

    shared = CUDA.@cuStaticSharedMem(Float32, 1024)
    shared[tx + Int32(1)] = local_sum
    shared[tx + Int32(513)] = local_sq
    sync_threads()

    stride = blockDim().x ÷ Int32(2)
    while stride > 0
        if tx < stride
            shared[tx + Int32(1)] += shared[tx + stride + Int32(1)]
            shared[tx + Int32(513)] += shared[tx + stride + Int32(513)]
        end
        sync_threads()
        stride ÷= Int32(2)
    end

    if tx == Int32(0)
        count = Float32(total)
        m = shared[1] / count
        mean[fid + Int32(1)] = m
        var[fid + Int32(1)] = shared[513] / count - m * m
    end
    return
end

function verify_kernel!(mean, var, flag, bs::Int32, fs::Int32, ss::Int32)
    fid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if fid < fs
        sum = 0.0f0
        sumsq = 0.0f0
        bid = Int32(0)
        while bid < bs
            s = Int32(0)
            while s < ss
                idx = Int64(bid) * Int64(ss) * Int64(fs) + Int64(fid) * Int64(ss) + Int64(s)
                v = input_value(idx)
                sum += v
                sumsq += v * v
                s += Int32(1)
            end
            bid += Int32(1)
        end
        count = Float32(bs * ss)
        ref_mean = sum / count
        ref_var = sumsq / count - ref_mean * ref_mean
        if abs(mean[fid + Int32(1)] - ref_mean) > 1.0f-3 ||
           abs(var[fid + Int32(1)] - ref_var) > 1.0f-3
            flag[1] = Int32(1)
        end
    end
    return
end

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function main(args)
    if length(args) != 4
        println("Usage: main.jl <batch_size> <spatial_size> <feature_size> <repeat>")
        return 1
    end

    bs = parse(Int32, args[1])
    ss = parse(Int32, args[2])
    fs = parse(Int32, args[3])
    repeat = parse(Int, args[4])

    total = Int64(bs) * Int64(ss) * Int64(fs)
    input = CuArray{Float32}(undef, total)
    mean = CuArray{Float32}(undef, Int(fs))
    var = CuArray{Float32}(undef, Int(fs))

    init_blocks = min(ceil_div(total, Int64(256)), Int64(65535))
    @cuda threads=256 blocks=Int(init_blocks) init_input!(input, total)
    CUDA.synchronize()

    @cuda threads=512 blocks=Int(fs) welford_kernel!(input, mean, var, bs, fs, ss)
    CUDA.synchronize()

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=512 blocks=Int(fs) welford_kernel!(input, mean, var, bs, fs, ss)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6
    @printf("Average kernel execution time %f (ms)\n", elapsed_ms / repeat)

    flag = CuArray([Int32(0)])
    @cuda threads=256 blocks=ceil_div(Int(fs), 256) verify_kernel!(mean, var, flag, bs, fs, ss)
    CUDA.synchronize()
    ok = CUDA.@allowscalar flag[1] == Int32(0)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
