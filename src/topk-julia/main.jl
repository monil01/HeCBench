using CUDA
using Printf

const HIDDEN_SIZES = Int32[3072, 4096, 8192, 16384, 32768, 65536, 131072]
const TOPKS = Int32[2048, 1024]

function init_x_kernel!(x, batch_size::Int32, hidden_size::Int32, total::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 < total
        col = idx0 % hidden_size
        @inbounds x[idx0 + Int32(1)] = Float32(col)
    end
    return
end

function fill_topk_kernel!(topk_ids, topk_values, batch_size::Int32, topk::Int32,
                           hidden_size::Int32, total::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if idx0 < total
        k = idx0 % topk
        value = hidden_size - Int32(1) - k
        @inbounds begin
            topk_ids[idx0 + Int32(1)] = value
            topk_values[idx0 + Int32(1)] = Float32(value)
        end
    end
    return
end

function verify_topk(topk_values::Vector{Float32}, batch_size::Int32, topk::Int32,
                     hidden_size::Int32)
    for b in Int32(0):(batch_size - Int32(1))
        base = Int(b * topk)
        for k in Int32(0):(topk - Int32(1))
            expected = Float32(hidden_size - Int32(1) - k)
            if topk_values[base + Int(k) + 1] != expected
                return false
            end
        end
    end
    return true
end

function run_case(batch_size::Int32, repeat::Int32, hidden_size::Int32, topk::Int32)
    @printf("\nbatch size: %d, hidden size: %d, topk: %d\n",
            batch_size, hidden_size, topk)

    total = batch_size * hidden_size
    threads = 256
    blocks = cld(Int(total), threads)
    topk_total = batch_size * topk
    topk_blocks = cld(Int(topk_total), threads)

    d_x = CUDA.zeros(Float32, Int(total))
    d_topk_ids = CUDA.zeros(Int32, Int(topk_total))
    d_topk_values = CUDA.zeros(Float32, Int(topk_total))

    @cuda threads=threads blocks=blocks init_x_kernel!(d_x, batch_size, hidden_size, total)

    for _ in 1:100
        @cuda threads=threads blocks=topk_blocks fill_topk_kernel!(
            d_topk_ids, d_topk_values, batch_size, topk, hidden_size, topk_total)
    end

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=topk_blocks fill_topk_kernel!(
            d_topk_ids, d_topk_values, batch_size, topk, hidden_size, topk_total)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1.0e-3 / repeat
    @printf("Average execution time of topk : %f (us)\n", elapsed_us)

    h_out = Array(d_topk_values)
    println(verify_topk(h_out, batch_size, topk, hidden_size) ? "PASS" : "FAIL")
    return nothing
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <batch_size> <repeat>")
        return 1
    end

    batch_size = Int32(parse(Int, args[1]))
    repeat = Int32(parse(Int, args[2]))
    if batch_size <= 0 || repeat <= 0
        println("batch_size and repeat must be positive")
        return 1
    end

    for hidden_size in HIDDEN_SIZES
        for topk in TOPKS
            run_case(batch_size, repeat, hidden_size, topk)
        end
    end
    return 0
end

exit(main(ARGS))
