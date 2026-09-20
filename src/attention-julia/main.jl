using CUDA
using Printf

function init_kvq!(key, value, query, total::Int64, d::Int32)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        col = Int32(i % Int64(d))
        key[i + 1] = (Float32((i * Int64(17) + Int64(3)) % Int64(2001)) - 1000.0f0) * 1.0f-6
        value[i + 1] = (Float32((i * Int64(29) + Int64(7)) % Int64(2001)) - 1000.0f0) * 1.0f-6
        if i < Int64(d)
            query[i + 1] = (Float32((Int64(col) * Int64(31) + Int64(11)) % Int64(2001)) - 1000.0f0) * 1.0f-6
        end
        i += stride
    end
    return
end

function attention_kernel1!(key, query, dot_product, exp_sum, n::Int32, d::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < n
        sum = 0.0f0
        j = Int32(0)
        while j < d
            sum += key[Int64(i) * Int64(d) + Int64(j) + 1] * query[j + Int32(1)]
            j += Int32(1)
        end
        dot_product[i + Int32(1)] = sum
        CUDA.@atomic exp_sum[1] += exp(sum)
    end
    return
end

function attention_kernel2!(exp_sum, dot_product, score, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < n
        score[i + Int32(1)] = exp(dot_product[i + Int32(1)]) / exp_sum[1]
    end
    return
end

function attention_kernel3!(score, value, output, n::Int32, d::Int32)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if j < d
        sum = 0.0f0
        i = Int32(0)
        while i < n
            sum += score[i + Int32(1)] * value[Int64(i) * Int64(d) + Int64(j) + 1]
            i += Int32(1)
        end
        output[j + Int32(1)] = sum
    end
    return
end

function verify_kernel!(out, ref, flag, d::Int32)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if j < d
        if abs(out[j + Int32(1)] - ref[j + Int32(1)]) > 1.0f-3
            flag[1] = Int32(1)
        end
    end
    return
end

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function attention_run!(key, value, query, dot_product, exp_sum, score, output,
                        n::Int32, d::Int32)
    fill!(exp_sum, 0.0f0)
    @cuda threads=256 blocks=ceil_div(Int(n), 256) attention_kernel1!(key, query, dot_product, exp_sum, n, d)
    @cuda threads=256 blocks=ceil_div(Int(n), 256) attention_kernel2!(exp_sum, dot_product, score, n)
    @cuda threads=256 blocks=ceil_div(Int(d), 256) attention_kernel3!(score, value, output, n, d)
    return
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <rows> <columns> <implementation> <repeat>")
        println("implementation 0: naive")
        println("implementation 1: fused kernels with block reduce")
        println("implementation 2: fused kernels with warp reduce")
        println("implementation 3: fused kernels with mixed reduce")
        return 1
    end

    n = parse(Int32, args[1])
    d = parse(Int32, args[2])
    impl_num = parse(Int, args[3])
    repeat = parse(Int, args[4])

    total = Int64(n) * Int64(d)
    key = CuArray{Float32}(undef, total)
    value = CuArray{Float32}(undef, total)
    query = CuArray{Float32}(undef, Int(d))
    blocks_init = min(ceil_div(total, Int64(256)), Int64(65535))
    @cuda threads=256 blocks=Int(blocks_init) init_kvq!(key, value, query, total, d)

    dot_product = CuArray{Float32}(undef, Int(n))
    exp_sum = CuArray([0.0f0])
    score = CuArray{Float32}(undef, Int(n))
    output = CuArray{Float32}(undef, Int(d))
    reference = CuArray{Float32}(undef, Int(d))

    attention_run!(key, value, query, dot_product, exp_sum, score, reference, n, d)
    CUDA.synchronize()

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        attention_run!(key, value, query, dot_product, exp_sum, score, output, n, d)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1.0e-6
    @printf("Average execution time of kernels %f (ms)\n", elapsed_ms / repeat)

    flag = CuArray([Int32(0)])
    @cuda threads=256 blocks=ceil_div(Int(d), 256) verify_kernel!(output, reference, flag, d)
    CUDA.synchronize()
    ok = CUDA.@allowscalar flag[1] == Int32(0)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
