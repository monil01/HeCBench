using CUDA
using LinearAlgebra
using Printf
using Random

const THREAD_BLOCK_SIZE = 1024

function preprocess!(data::Matrix{Float32})
    n, l = size(data)
    for i in 1:n
        row = @view data[i, :]
        mean = sum(Float64, row) / l
        denom = sqrt(sum((Float64(x) - mean)^2 for x in row))
        if denom == 0
            fill!(row, 0f0)
        else
            for j in 1:l
                row[j] = Float32((Float64(row[j]) - mean) / denom)
            end
        end
    end
    return data
end

function extract_upper_kernel!(cormat, upper, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = n * n
    if idx <= total
        z = idx - Int32(1)
        i = z ÷ n
        j = z - i * n
        if i < j
            t = n * i - (i * (i + Int32(1))) ÷ Int32(2) + j - i
            upper[t] = cormat[j + Int32(1), i + Int32(1)]
        end
    end
    return
end

function upper_cpu(data::Matrix{Float32})
    n, _ = size(data)
    upper = Vector{Float32}(undef, n * (n - 1) ÷ 2)
    p = 1
    for i in 1:n-1
        for j in i+1:n
            upper[p] = dot(@view(data[i, :]), @view(data[j, :]))
            p += 1
        end
    end
    return upper
end

function parse_args()
    length(ARGS) == 2 || begin
        println("Usage: main.jl <Number of voxels> <Length of time series>")
        exit(1)
    end
    return parse(Int, ARGS[1]), parse(Int, ARGS[2])
end

function main()
    n, l = parse_args()
    println("Number of voxels: $n  Length of time series: $l\n")

    rng = MersenneTwister(123)
    data = Float32.(rand(rng, Float32, n, l) .* 12f0 .- 6f0)
    preprocess!(data)
    m11 = n * (n - 1) ÷ 2

    println("\nComputing correlations ...")

    d_data = CuArray(data')
    CUDA.synchronize()
    t0 = time_ns()
    d_cormat = d_data' * d_data
    CUDA.synchronize()
    gemm_ns = time_ns() - t0

    d_upper = CUDA.zeros(Float32, m11)
    total = n * n
    blocks = cld(total, THREAD_BLOCK_SIZE)
    CUDA.synchronize()
    t1 = time_ns()
    @cuda threads=THREAD_BLOCK_SIZE blocks=blocks extract_upper_kernel!(d_cormat, d_upper, Int32(n))
    CUDA.synchronize()
    extract_ns = time_ns() - t1

    upper = Array(d_upper)
    @printf("Kernel time (s)\nGEMM: %.9f, Extract upper triangle: %.9f\n",
            gemm_ns * 1e-9, extract_ns * 1e-9)
    @printf("\nRunning time for computing correlations: \n%.9f (s)\n",
            (gemm_ns + extract_ns) * 1e-9)

    checksum = sum(Float64, upper)
    println("Checksum: $checksum")

    ref = upper_cpu(data)
    max_err = maximum(abs.(Float64.(upper) .- Float64.(ref)); init=0.0)
    println(max_err <= 1e-3 * max(1.0, maximum(abs.(Float64.(ref)); init=0.0)) ? "PASS" : "FAIL")
end

main()
