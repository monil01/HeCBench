using CUDA
using Printf
using Random

const BLOCK_DIM = 64
const N = 1_999_999

@inline function round5(x::Float32)
    return round(x * Float32(1f5)) / Float32(1f5)
end

@inline cbrtf_host(x::Float32) = cbrt(x)

function generate_data(size, minv, maxv)
    rng = MersenneTwister(1993764)
    return Float32.(rand(rng, minv:maxv, size))
end

function quartic_minimum_cpu!(A, B, C, D, minimum)
    @inbounds for i in eachindex(A)
        b = Float32(0.75) * (B[i] / A[i])
        c = Float32(0.50) * (C[i] / A[i])
        d = Float32(0.25) * (D[i] / A[i])
        q = round5(c / Float32(3) - (b * b) / Float32(9))
        r = round5((b * c) / Float32(6) - (b * b * b) / Float32(27) - Float32(0.5) * d)
        del = r * r + q * q * q
        if del <= Float32(1f-5)
            theta = acos(r / sqrt(-(q * q * q)))
            sqrtq = Float32(2) * sqrt(-q)
            x1 = sqrtq * cos(theta / Float32(3)) - b / Float32(3)
            x2 = sqrtq * cos((theta + Float32(2) * Float32(3.1415927)) / Float32(3)) - b / Float32(3)
            x3 = sqrtq * cos((theta + Float32(4) * Float32(3.1415927)) / Float32(3)) - b / Float32(3)
            if x1 < x2
                x1, x2 = x2, x1
            end
            if x2 < x3
                x2, x3 = x3, x2
            end
            if x1 < x2
                x1, x2 = x2, x1
            end
            delta = A[i] * ((x1^4 - x3^4) / Float32(4)) +
                    B[i] * ((x1^3 - x3^3) / Float32(3)) +
                    C[i] * ((x1^2 - x3^2) / Float32(2)) +
                    D[i] * (x1 - x3)
            minimum[i] = delta <= 0 ? x1 : x3
        else
            minimum[i] = cbrtf_host(r + sqrt(del)) + cbrtf_host(r - sqrt(del)) - b / Float32(3)
        end
    end
end

function qrdel_kernel!(n::Int32, A, B, C, D, b, c, d, Q, R, Qint, Rint, del)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            b[i] = Float32(0.75) * (B[i] / A[i])
            c[i] = Float32(0.50) * (C[i] / A[i])
            d[i] = Float32(0.25) * (D[i] / A[i])
            Q[i] = round((c[i] / Float32(3) - (b[i] * b[i]) / Float32(9)) * Float32(1f5)) / Float32(1f5)
            R[i] = round(((b[i] * c[i]) / Float32(6) - (b[i] * b[i] * b[i]) / Float32(27) - Float32(0.5) * d[i]) * Float32(1f5)) / Float32(1f5)
            Qint[i] = Q[i] * Q[i] * Q[i]
            Rint[i] = R[i] * R[i]
            del[i] = Rint[i] + Qint[i]
        end
    end
    return
end

@inline cbrtf_device(x::Float32) = cbrt(x)

function solver_kernel!(n::Int32, A, B, C, D, b, Q, R, del, theta, sqrtQ, x1, x2, x3, temp, minimum)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            if del[i] <= Float32(1f-5)
                theta[i] = acos(R[i] / sqrt(-(Q[i] * Q[i] * Q[i])))
                sqrtQ[i] = Float32(2) * sqrt(-Q[i])
                x1[i] = sqrtQ[i] * cos(theta[i] / Float32(3)) - b[i] / Float32(3)
                x2[i] = sqrtQ[i] * cos((theta[i] + Float32(2) * Float32(3.1415927)) / Float32(3)) - b[i] / Float32(3)
                x3[i] = sqrtQ[i] * cos((theta[i] + Float32(4) * Float32(3.1415927)) / Float32(3)) - b[i] / Float32(3)
                if x1[i] < x2[i]
                    temp[i] = x1[i]
                    x1[i] = x2[i]
                    x2[i] = temp[i]
                end
                if x2[i] < x3[i]
                    temp[i] = x2[i]
                    x2[i] = x3[i]
                    x3[i] = temp[i]
                end
                if x1[i] < x2[i]
                    temp[i] = x1[i]
                    x1[i] = x2[i]
                    x2[i] = temp[i]
                end
                delta = A[i] * ((x1[i]^4 - x3[i]^4) / Float32(4)) +
                        B[i] * ((x1[i]^3 - x3[i]^3) / Float32(3)) +
                        C[i] * ((x1[i]^2 - x3[i]^2) / Float32(2)) +
                        D[i] * (x1[i] - x3[i])
                minimum[i] = delta <= 0 ? x1[i] : x3[i]
            else
                x1[i] = cbrtf_device(R[i] + sqrt(del[i])) + cbrtf_device(R[i] - sqrt(del[i])) - b[i] / Float32(3)
                x2[i] = Float32(0)
                x3[i] = Float32(0)
                minimum[i] = x1[i]
            end
        end
    end
    return
end

function quartic_minimum_gpu!(A, B, C, D, minimum)
    n = length(A)
    d_A, d_B, d_C, d_D = CuArray(A), CuArray(B), CuArray(C), CuArray(D)
    d_b = CUDA.zeros(Float32, n)
    d_c = CUDA.zeros(Float32, n)
    d_d = CUDA.zeros(Float32, n)
    d_theta = CUDA.zeros(Float32, n)
    d_sqrtQ = CUDA.zeros(Float32, n)
    d_Q = CUDA.zeros(Float32, n)
    d_R = CUDA.zeros(Float32, n)
    d_Qint = CUDA.zeros(Float32, n)
    d_Rint = CUDA.zeros(Float32, n)
    d_del = CUDA.zeros(Float32, n)
    d_x1 = CUDA.zeros(Float32, n)
    d_x2 = CUDA.zeros(Float32, n)
    d_x3 = CUDA.zeros(Float32, n)
    d_temp = CUDA.zeros(Float32, n)
    d_min = CUDA.zeros(Float32, n)
    blocks = cld(n, BLOCK_DIM)
    @cuda threads=BLOCK_DIM blocks=blocks qrdel_kernel!(
        Int32(n), d_A, d_B, d_C, d_D, d_b, d_c, d_d, d_Q, d_R, d_Qint, d_Rint, d_del)
    @cuda threads=BLOCK_DIM blocks=blocks solver_kernel!(
        Int32(n), d_A, d_B, d_C, d_D, d_b, d_Q, d_R, d_del, d_theta, d_sqrtQ,
        d_x1, d_x2, d_x3, d_temp, d_min)
    CUDA.synchronize()
    copyto!(minimum, d_min)
end

function matches_cuda_tolerance(actual, reference; tol=Float32(1f-3))
    @inbounds for i in eachindex(actual, reference)
        diff = abs(actual[i] - reference[i])
        if diff > tol
            return false
        end
    end
    return true
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        exit(1)
    end
    repeat = parse(Int, ARGS[1])
    println("N = $N")

    A = generate_data(N, -100, 100)
    B = generate_data(N, -100, 100)
    C = generate_data(N, -100, 100)
    D = generate_data(N, -100, 100)
    E = generate_data(N, -100, 100)
    @inbounds for i in eachindex(A)
        if A[i] == 0
            A[i] = 1
        end
    end
    minimum_ref = Vector{Float32}(undef, N)
    minimum = Vector{Float32}(undef, N)

    println("generating data...")
    println("####################### Reference #############")
    start = time()
    quartic_minimum_cpu!(A, B, C, D, minimum_ref)
    ref_ms = (time() - start) * 1000
    @printf("Execution time (ms): %f\n", ref_ms)

    println("####################### GPU (no streams) #############")
    avg = 0.0
    for _ in 1:repeat
        start = time()
        quartic_minimum_gpu!(A, B, C, D, minimum)
        avg += (time() - start) * 1000
    end
    @printf("Execution time (ms): %f\n", avg / repeat)
    ok = matches_cuda_tolerance(minimum, minimum_ref)
    println(ok ? "PASS" : "FAIL")

    println("####################### GPU (streams) #############")
    avg = 0.0
    for _ in 1:repeat
        start = time()
        quartic_minimum_gpu!(A, B, C, D, minimum)
        avg += (time() - start) * 1000
    end
    @printf("Execution time (ms): %f\n", avg / repeat)
    ok = matches_cuda_tolerance(minimum, minimum_ref)
    println(ok ? "PASS" : "FAIL")
end

main()
