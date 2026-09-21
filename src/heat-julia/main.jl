using CUDA
using Printf

const LINE = "--------------------"

function initial_value_kernel!(u, n::Int32, dx::Float64, length::Float64)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = n * n
    if idx0 < total
        i = idx0 % n
        j = idx0 ÷ n
        y = dx * Float64(j + Int32(1))
        x = dx * Float64(i + Int32(1))
        @inbounds u[idx0 + Int32(1)] = sin(pi * x / length) * sin(pi * y / length)
    end
    return
end

function solve_kernel!(u_tmp, u, n::Int32, r::Float64, r2::Float64)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = n * n
    if idx0 < total
        i = idx0 % n
        j = idx0 ÷ n
        center = @inbounds u[idx0 + Int32(1)]
        right = i < n - Int32(1) ? (@inbounds u[idx0 + Int32(2)]) : 0.0
        left = i > Int32(0) ? (@inbounds u[idx0]) : 0.0
        down = j < n - Int32(1) ? (@inbounds u[idx0 + n + Int32(1)]) : 0.0
        up = j > Int32(0) ? (@inbounds u[idx0 - n + Int32(1)]) : 0.0
        @inbounds u_tmp[idx0 + Int32(1)] = r2 * center + r * (right + left + down + up)
    end
    return
end

solution(t, x, y, alpha, length) =
    exp(-2.0 * alpha * pi * pi * t / (length * length)) * sin(pi * x / length) * sin(pi * y / length)

function l2norm(n::Int, u::Vector{Float64}, nsteps::Int, dt::Float64,
                alpha::Float64, dx::Float64, length::Float64)
    time = dt * nsteps
    acc = 0.0
    y = dx
    for j in 0:n-1
        x = dx
        for i in 0:n-1
            answer = solution(time, x, y, alpha, length)
            diff = u[i + j * n + 1] - answer
            acc += diff * diff
            x += dx
        end
        y += dx
    end
    return sqrt(acc)
end

function main()
    start_total = time()
    n = 1000
    nsteps = 10
    if length(ARGS) == 2
        n = parse(Int, ARGS[1])
        nsteps = parse(Int, ARGS[2])
        if n < 0 || nsteps < 0
            println(stderr, "Error: n and nsteps must be positive")
            return 1
        end
    end

    alpha = 0.1
    domain_length = 1000.0
    dx = domain_length / (n + 1)
    dt = 0.5 / nsteps
    r = alpha * dt / (dx * dx)
    r2 = 1.0 - 4.0 * r

    device_name = CUDA.name(CUDA.device())
    println()
    println(" MMS heat equation")
    println()
    println(LINE)
    println("Problem input")
    println()
    println(" Grid size: $n x $n")
    println(" Cell width: $dx")
    println(" Grid length: $(domain_length)x$(domain_length)")
    println()
    println(" Alpha: $alpha")
    println()
    println(" Steps: $nsteps")
    println(" Total time: $(dt * nsteps)")
    println(" Time step: $dt")
    println(" GPU device: $device_name")
    println(LINE)
    println("Stability")
    println()
    println(" r value: $r")
    if r > 0.5
        println(" Warning: unstable")
    end
    println(LINE)

    total = n * n
    u = CUDA.zeros(Float64, total)
    u_tmp = CUDA.zeros(Float64, total)
    block_size = 256
    grid = cld(total, block_size)
    @cuda threads=block_size blocks=grid initial_value_kernel!(u, Int32(n), dx, domain_length)
    CUDA.synchronize()

    tic = time()
    for _ in 1:nsteps
        @cuda threads=block_size blocks=grid solve_kernel!(u_tmp, u, Int32(n), r, r2)
        u, u_tmp = u_tmp, u
    end
    CUDA.synchronize()
    toc = time()

    u_host = Array(u)
    norm = l2norm(n, u_host, nsteps, dt, alpha, dx, domain_length)
    stop_total = time()
    solve_time = toc - tic
    total_time = stop_total - start_total
    bandwidth = 1.0e-9 * 2.0 * n * n * nsteps * sizeof(Float64) / solve_time

    println("Results")
    println()
    println("Error (L2norm): $norm")
    println("Solve time (s): $solve_time")
    println("Total time (s): $total_time")
    println("Bandwidth (GB/s): $bandwidth")
    println(LINE)
    return 0
end

exit(main())
