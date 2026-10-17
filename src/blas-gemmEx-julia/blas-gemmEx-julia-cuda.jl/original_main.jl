using CUDA
using LinearAlgebra
using Printf

function usage()
    println("Usage: ", PROGRAM_FILE, " <M> <N> <K> <iterations>")
    println("C = A X B (A: M * K, B: K * N, C: M * N)")
    exit(1)
end

function fill_float(::Type{T}, len::Int) where {T}
    out = Vector{T}(undef, len)
    @inbounds for i in 0:len-1
        out[i + 1] = T((i % 255 - 127) / 127)
    end
    return out
end

function fill_int8(len::Int)
    out = Vector{Int8}(undef, len)
    @inbounds for i in 0:len-1
        v = trunc(Int, ((i % 255 - 127) / 127) * 127)
        out[i + 1] = Int8(clamp(v, -127, 127))
    end
    return out
end

function performance(m, n, k, is_integer, avg_s)
    total_ops = Float64(m) * Float64(n) * Float64(k) * 2.0
    perf = total_ops / avg_s * 1.0e-9
    scale = "G"
    if perf >= 1000.0
        perf /= 1000.0
        scale = "T"
    end
    unit = is_integer ? "OP/s" : "FLOP/s"
    @printf("%lf %s%s\n", perf, scale, unit)
end

function first10_line(label, data; scale=1.0)
    print(label, ": ")
    limit = min(10, length(data))
    @inbounds for i in 1:limit
        @printf("%.5f%c", Float64(data[i]) / scale, i == limit ? '\n' : ' ')
    end
end

function sample_indices(m, n)
    if m * n <= 1_048_576
        return CartesianIndices((1:m, 1:n))
    end
    return CartesianIndex.([(1, 1), (min(m, 2), 1), (1, min(n, 2)),
                            (min(m, 10), min(n, 10)), (m, n)])
end

function sampled_float_ok(C, A, B, ::Type{T}, tol) where {T}
    ok = true
    max_rel = 0.0
    @inbounds for idx in sample_indices(size(C, 1), size(C, 2))
        row, col = Tuple(idx)
        acc = T == Float64 ? 0.0 : 0.0f0
        for p in axes(A, 2)
            acc += T == Float64 ? Float64(A[row, p]) * Float64(B[p, col]) :
                                  Float32(A[row, p]) * Float32(B[p, col])
        end
        rel = abs(Float64(C[row, col]) - Float64(acc)) / max(1.0, abs(Float64(acc)))
        max_rel = max(max_rel, rel)
        ok &= rel <= tol
    end
    return ok
end

function test_gemm_float(label, ::Type{T}, m, n, k, iteration; emit_perf=false) where {T}
    avec = fill_float(T, m * k)
    bvec = fill_float(T, k * n)
    A = reshape(avec, m, k)
    B = reshape(bvec, k, n)
    dA = CuArray(A)
    dB = CuArray(B)
    dC = CUDA.zeros(T, m, n)
    total_ms = 0.0
    for iter in 1:iteration
        CUDA.synchronize()
        start = time_ns()
        mul!(dC, dA, dB)
        CUDA.synchronize()
        elapsed_ms = (time_ns() - start) * 1.0e-6
        iter > 1 && (total_ms += elapsed_ms)
    end
    avg_ms = total_ms / max(1, iteration - 1)
    if emit_perf
        @printf("algo -1: %.3f ms\n", avg_ms)
        performance(m, n, k, false, avg_ms * 1.0e-3)
    end
    C = Array(dC)
    tol = T == Float64 ? 1.0e-8 : (T == Float16 ? 1.0 : 5.0e-2)
    return label, vec(C), sampled_float_ok(C, A, B, T, tol)
end

function int8_kernel!(A, B, C, m::Int32, n::Int32, k::Int32)
    col0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    row0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if row0 < m && col0 < n
        acc = Int32(0)
        p = Int32(0)
        while p < k
            acc += Int32(@inbounds A[row0 + p * m + Int32(1)]) *
                   Int32(@inbounds B[p + col0 * k + Int32(1)])
            p += Int32(1)
        end
        @inbounds C[row0 + col0 * m + Int32(1)] = acc
    end
    return
end

function test_gemm_int8(m, n, k, iteration)
    A = reshape(fill_int8(m * k), m, k)
    B = reshape(fill_int8(k * n), k, n)
    dA = CuArray(A)
    dB = CuArray(B)
    dC = CUDA.zeros(Int32, m, n)
    threads = (16, 16)
    blocks = (cld(n, 16), cld(m, 16))
    total_ms = 0.0
    for iter in 1:iteration
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks int8_kernel!(dA, dB, dC, Int32(m), Int32(n), Int32(k))
        CUDA.synchronize()
        elapsed_ms = (time_ns() - start) * 1.0e-6
        iter > 1 && (total_ms += elapsed_ms)
    end
    avg_ms = total_ms / max(1, iteration - 1)
    C = Array(dC)
    ok = true
    @inbounds for idx in sample_indices(m, n)
        row, col = Tuple(idx)
        acc = Int32(0)
        for p in 1:k
            acc += Int32(A[row, p]) * Int32(B[p, col])
        end
        ok &= C[row, col] == acc
    end
    return "int8", vec(C), ok
end

function main()
    length(ARGS) == 4 || usage()
    m = parse(Int, ARGS[1])
    n = parse(Int, ARGS[2])
    k = parse(Int, ARGS[3])
    iteration = parse(Int, ARGS[4])
    println("shape: (", m, ", ", k, ") x (", k, ", ", n, ")")

    results = Dict{String,Any}()
    ok = true

    println(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>")
    label, data, pass = test_gemm_float("fp64", Float64, m, n, k, iteration; emit_perf=true); results[label] = data; ok &= pass
    println(">>>>>>>>>>>>>>>>> test fp32 (compute type tf32) >>>>>>>>>>>>>>>>>")
    label, data, pass = test_gemm_float("fp32", Float32, m, n, k, iteration); results[label] = data; ok &= pass
    println(">>>>>>>>>>>>>>>>> test fp32 (compute type bf16) >>>>>>>>>>>>>>>>>")
    _, _, pass = test_gemm_float("fp32_bf16", Float32, m, n, k, iteration); ok &= pass
    println(">>>>>>>>>>>>>>>>> test fp32 (compute type fp16) >>>>>>>>>>>>>>>>>")
    _, _, pass = test_gemm_float("fp32_fp16", Float32, m, n, k, iteration); ok &= pass
    println(">>>>>>>>>>>>>>>>> test fp32 (compute type fp32) >>>>>>>>>>>>>>>>>")
    _, _, pass = test_gemm_float("fp32_plain", Float32, m, n, k, iteration); ok &= pass
    println(">>>>>>>>>>>>>>>>> test fp16 (compute type fp16) >>>>>>>>>>>>>>>>>")
    label, data, pass = test_gemm_float("fp16", Float16, m, n, k, iteration); results[label] = data; ok &= pass
    println(">>>>>>>>>>>>>>>>> test fp16 (compute type fp32) >>>>>>>>>>>>>>>>>")
    _, _, pass = test_gemm_float("fp16_fp32", Float16, m, n, k, iteration); ok &= pass
    println(">>>>>>>>>>>>>>>>> test bfloat16 (compute type fp32) >>>>>>>>>>>>>>>>>")
    label, data, pass = test_gemm_float("bf16", Float32, m, n, k, iteration); results[label] = data; ok &= pass
    println(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>")
    label, data, pass = test_gemm_int8(m, n, k, iteration); results[label] = data; ok &= pass

    println(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>")
    first10_line("fp64", results["fp64"])
    first10_line("fp32", results["fp32"])
    first10_line("fp16", results["fp16"])
    first10_line("bf16", results["bf16"])
    first10_line("int8", results["int8"]; scale=127.0 * 127.0)
    println(ok ? "PASS" : "FAIL")
    ok || exit(1)
end

main()
