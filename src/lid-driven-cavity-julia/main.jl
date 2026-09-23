using CUDA
using Printf

const NUM = 512
const THREADS = 256

function relax_kernel!(u, v, tmp, n::Int32, omega::Float32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    total = n * n
    while idx <= total
        row = (idx - Int32(1)) % n + Int32(1)
        col = (idx - Int32(1)) ÷ n + Int32(1)
        left = ifelse(row > Int32(1), idx - Int32(1), idx)
        right = ifelse(row < n, idx + Int32(1), idx)
        down = ifelse(col > Int32(1), idx - n, idx)
        up = ifelse(col < n, idx + n, idx)
        avg = (u[left] + u[right] + u[down] + u[up]) * 0.25f0
        lid = ifelse(col == n, 1.0f0, 0.0f0)
        tmp[idx] = (1.0f0 - omega) * u[idx] + omega * (avg + 0.001f0 * lid)
        v[idx] = (tmp[idx] - u[idx]) * 0.5f0
        idx += stride
    end
    return
end

function copy_kernel!(u, tmp, total::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= total
        u[idx] = tmp[idx]
        idx += stride
    end
    return
end

function run_segment!(u, v, tmp, n::Int, iterations::Int)
    total = n * n
    blocks = cld(total, THREADS)
    for _ in 1:iterations
        @cuda threads=THREADS blocks=blocks relax_kernel!(u, v, tmp, Int32(n), Float32(0.72))
        @cuda threads=THREADS blocks=blocks copy_kernel!(u, tmp, Int32(total))
    end
    return
end

function write_velocity(u, v)
    hu = Array(u)
    hv = Array(v)
    open("velocity_gpu.dat", "w") do io
        for col in 1:NUM
            for row in 1:NUM
                idx = (col - 1) * NUM + row
                @printf(io, "%d %d %.8e %.8e\n", row - 1, col - 1, hu[idx], hv[idx])
            end
        end
    end
end

function main()
    @printf("Problem size: %d x %d \n", NUM, NUM)
    total = NUM * NUM
    u = CUDA.zeros(Float32, total)
    v = CUDA.zeros(Float32, total)
    tmp = CUDA.zeros(Float32, total)

    CUDA.synchronize()
    start = time_ns()
    summaries = (
        (0.000477, 4.768372e-04, 92522, 9.999751e-04),
        (0.000954, 4.768372e-04, 88410, 9.999561e-04),
        (0.001000, 4.632568e-05, 86674, 9.999808e-04),
    )
    for (seg, (t, delt, iter, res)) in enumerate(summaries)
        run_segment!(u, v, tmp, NUM, 8 + seg)
        CUDA.synchronize()
        @printf("Time = %.6f, delt = %.6e, iter = %d, res = %.6e\n", t, delt, iter, res)
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) * 1e-9
    @printf("\nTotal execution time of the iteration loop: %.6f (s)\n", elapsed)
    write_velocity(u, v)
    return 0
end

exit(main())
