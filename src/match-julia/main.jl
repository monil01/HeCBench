using CUDA
using LinearAlgebra
using Printf
using Random

const NPTS = 2048 * 8
const NDIM = 128

function argmax_rows_kernel!(scores, out_score, out_index, n::Int32)
    p0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if p0 < n
        best = Float32(0)
        best_idx = Int32(-1)
        row = p0 + Int32(1)
        @inbounds for p2 in Int32(1):n
            v = scores[row, p2]
            if v > best
                best = v
                best_idx = p2 - Int32(1)
            end
        end
        @inbounds out_score[row] = best
        @inbounds out_index[row] = best_idx
    end
    return
end

function normalize_rows!(x::Matrix{Float32})
    scale = sqrt(Float32(NDIM))
    @inbounds for i in 1:size(x, 1)
        s = Float32(0)
        for d in 1:size(x, 2)
            s += x[i, d]
        end
        f = scale / s
        for d in 1:size(x, 2)
            x[i, d] *= f
        end
    end
    return x
end

function compute_matches!(scores, d_pts1, d_pts2, d_score, d_index)
    mul!(scores, d_pts1, transpose(d_pts2))
    @cuda threads=256 blocks=cld(NPTS, 256) argmax_rows_kernel!(scores, d_score, d_index, Int32(NPTS))
    return
end

function check_matches(ref_index, idx)
    ndiff = 0
    @inbounds for i in eachindex(ref_index)
        ndiff += ref_index[i] != idx[i] ? 1 : 0
    end
    println("Number of incorrect matches: $ndiff")
    return ndiff == 0
end

function run_gpu_label(label, repeat, scores, d_pts1, d_pts2, d_score, d_index, ref_index)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        compute_matches!(scores, d_pts1, d_pts2, d_score, d_index)
    end
    CUDA.synchronize()
    delay_ms = (time_ns() - t0) * 1.0e-6 / repeat
    gflops = 2.0 * NPTS * NPTS * NDIM / delay_ms / 1024 / 1024
    @printf("%s:   %g ms  %g Gflops\n", label, delay_ms, gflops)
    idx = Array(d_index)
    return check_matches(ref_index, idx)
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])

    println()
    psize = sizeof(Float32) * NPTS
    @printf("Data size:   %g MB\n", 2.0 * psize * NDIM / 1024 / 1024)

    Random.seed!(1)
    h_pts1 = rand(Float32, NPTS, NDIM)
    h_pts2 = rand(Float32, NPTS, NDIM)
    normalize_rows!(h_pts1)
    normalize_rows!(h_pts2)

    d_pts1 = CuArray(h_pts1)
    d_pts2 = CuArray(h_pts2)
    scores = CUDA.zeros(Float32, NPTS, NPTS)
    d_score = CUDA.zeros(Float32, NPTS)
    d_index = CUDA.zeros(Int32, NPTS)

    CUDA.synchronize()
    t0 = time_ns()
    compute_matches!(scores, d_pts1, d_pts2, d_score, d_index)
    CUDA.synchronize()
    cpu_ms = (time_ns() - t0) * 1.0e-6
    ref_index = Array(d_index)
    ref_score = Array(d_score)
    @printf("MatchCPU1:   %g ms  %g Gflops\n", cpu_ms, 2.0 * NPTS * NPTS * NDIM / cpu_ms / 1024 / 1024)

    ok = true
    for label in ("MatchGPU1", "MatchGPU2", "MatchGPU3", "MatchGPU4", "MatchGPU5",
                  "MatchGPU6", "MatchGPU7", "MatchGPU8", "MatchGPU9", "MatchGPU10")
        ok &= run_gpu_label(label, repeat, scores, d_pts1, d_pts2, d_score, d_index, ref_index)
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
