using CUDA
using Printf

# Julia port of jacobi-cuda benchmark. 2D Jacobi relaxation until L2 error
# falls below tolerance. Uses plain-global-memory 4-point stencil (simpler
# than the shared-mem tiled version) plus atomic error accumulation.

const NX = 2048

@inline idx_lin(i, j, N) = (i - 1) + (j - 1) * N + 1   # column-major 1-based

function jacobi_kernel!(f, f_old, err_arr, N::Int32)
    tx = Int32(threadIdx().x)  # 1..16
    ty = Int32(threadIdx().y)
    i = tx + Int32((blockIdx().x - 1)) * Int32(blockDim().x)  # 1..N
    j = ty + Int32((blockIdx().y - 1)) * Int32(blockDim().y)

    err = 0.0f0
    @inbounds if i >= Int32(2) && i <= N - Int32(1) && j >= Int32(2) && j <= N - Int32(1)
        c = (i - Int32(1)) + (j - Int32(1)) * N + Int32(1)
        ip = c + Int32(1)
        im = c - Int32(1)
        jp = c + N
        jm = c - N
        val = 0.25f0 * (f_old[ip] + f_old[im] + f_old[jp] + f_old[jm])
        f[c] = val
        df = val - f_old[c]
        err = df * df
    end
    # Simple: each thread with err > 0 atomic-adds. Fine for correctness.
    if err > 0.0f0
        CUDA.@atomic err_arr[1] += err
    end
    return
end

function initialize_data!(f, N)
    two_pi = 2 * pi
    for j in 1:N, i in 1:N
        if i == 1 || i == N
            f[i + (j - 1) * N] = Float32(sin((j - 1) * two_pi / (N - 1)))
        elseif j == 1 || j == N
            f[i + (j - 1) * N] = Float32(sin((i - 1) * two_pi / (N - 1)))
        else
            f[i + (j - 1) * N] = 0.0f0
        end
    end
end

function main()
    N = NX
    f = Vector{Float32}(undef, N * N)
    f_old = Vector{Float32}(undef, N * N)
    initialize_data!(f, N)
    initialize_data!(f_old, N)

    d_f = CuArray(f)
    d_f_old = CuArray(f_old)
    d_err = CUDA.zeros(Float32, 1)

    error = typemax(Float32)
    tolerance = 1f-5
    max_iters = 10000
    num_iters = 0

    threads = (16, 16)
    blocks = (cld(N, 16), cld(N, 16))

    CUDA.synchronize()
    t0 = time_ns()

    while error > tolerance && num_iters < max_iters
        CUDA.fill!(d_err, 0f0)
        @cuda threads=threads blocks=blocks jacobi_kernel!(d_f, d_f_old, d_err, Int32(N))
        d_f, d_f_old = d_f_old, d_f
        e = CUDA.@allowscalar d_err[1]
        error = sqrt(e / (N * N))
        if num_iters % 1000 == 0
            @printf("Error after iteration %d = %g\n", num_iters, error)
        end
        num_iters += 1
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9 / max(num_iters, 1)
    @printf("Average execution time per iteration: %g (s)\n", elapsed_s)

    if error <= tolerance && num_iters < max_iters
        println("PASS")
    else
        println("FAIL")
    end
    return 0
end

main()
