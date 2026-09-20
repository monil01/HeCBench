using CUDA
using Printf

const TPB = Int32(256)
const COLS_PER_BLK = Int32(32)
const ROWS_PER_THREAD = Int32(4)

function init_data_kernel!(data, total::Int64)
    idx0 = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x) - 1
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    i = idx0
    while i < total
        data[i + 1] = Float32(i % 1024) / 1024.0f0
        i += stride
    end
    return
end

function sop_kernel!(std, data, D::Int32, N::Int32)
    rows_per_blk_per_iter = TPB ÷ COLS_PER_BLK
    tx0 = threadIdx().x - Int32(1)
    this_col = tx0 % COLS_PER_BLK
    this_row = tx0 ÷ COLS_PER_BLK
    col = this_col + (blockIdx().y - Int32(1)) * COLS_PER_BLK
    row = this_row + (blockIdx().x - Int32(1)) * rows_per_blk_per_iter
    stride = rows_per_blk_per_iter * gridDim().x

    thread_sum = 0.0f0
    r = row
    while r < N
        if col < D
            v = data[Int64(r) * Int64(D) + Int64(col) + 1]
            thread_sum += v * v
        end
        r += stride
    end

    sstd = CUDA.@cuStaticSharedMem(Float32, 32)
    if tx0 < COLS_PER_BLK
        sstd[tx0 + 1] = 0.0f0
    end
    sync_threads()

    CUDA.@atomic sstd[this_col + 1] += thread_sum
    sync_threads()

    if tx0 < COLS_PER_BLK && col < D
        CUDA.@atomic std[col + 1] += sstd[tx0 + 1]
    end
    return
end

function sample_kernel!(std, D::Int32, sample_size::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < D
        std[i + 1] = sqrt(std[i + 1] / Float32(sample_size))
    end
    return
end

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function stddev_gpu!(std, data, D::Int32, N::Int32, sample::Bool)
    fill!(std, 0.0f0)
    rows_per_blk = (TPB ÷ COLS_PER_BLK) * ROWS_PER_THREAD
    grid_x = ceil_div(Int(N), Int(rows_per_blk))
    grid_y = ceil_div(Int(D), Int(COLS_PER_BLK))
    @cuda threads=Int(TPB) blocks=(grid_x, grid_y) sop_kernel!(std, data, D, N)

    sample_size = sample ? N - Int32(1) : N
    final_blocks = ceil_div(Int(D), Int(TPB))
    @cuda threads=Int(TPB) blocks=final_blocks sample_kernel!(std, D, sample_size)
    return
end

function reference_value(col::Int, D::Int, N::Int, sample::Bool)
    sample_size = sample ? N - 1 : N
    if D % 1024 == 0
        v = Float32((col - 1) % 1024) / 1024.0f0
        return sqrt(Float32(N) * v * v / Float32(sample_size))
    end

    sumsq = 0.0f0
    for r in 0:(N - 1)
        v = Float32((Int64(r) * Int64(D) + Int64(col - 1)) % 1024) / 1024.0f0
        sumsq += v * v
    end
    return sqrt(sumsq / Float32(sample_size))
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <D> <N> <repeat>")
        println("D: number of columns of data (must be a multiple of 32)")
        println("N: number of rows of data (at least one row)")
        return 1
    end

    D = parse(Int32, args[1])
    N = parse(Int32, args[2])
    repeat = parse(Int, args[3])
    sample = true

    total = Int64(D) * Int64(N)
    data = CuArray{Float32}(undef, total)
    std = CuArray{Float32}(undef, Int(D))

    init_blocks = min(ceil_div(total, Int64(TPB)), Int64(65535))
    @cuda threads=Int(TPB) blocks=Int(init_blocks) init_data_kernel!(data, total)
    CUDA.synchronize()

    stddev_gpu!(std, data, D, N, sample)
    CUDA.synchronize()

    start = time_ns()
    for _ in 1:repeat
        stddev_gpu!(std, data, D, N, sample)
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9
    @printf("Average execution time of stddev kernels: %f (s)\n", elapsed_s / repeat)

    got = Array(std)
    ok = true
    for c in 1:Int(D)
        ref = reference_value(c, Int(D), Int(N), sample)
        if abs(got[c] - ref) > 1.0f-3
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
