using CUDA
using Printf

function i4_ceiling(x::Float64)
    value = Int32(trunc(x))
    if Float64(value) < x
        value += Int32(1)
    end
    return value
end

function potential_device(a::Float64, b::Float64, x::Float64, y::Float64)
    return 2.0 * ((x / a / a)^2.0 + (y / b / b)^2.0) + 1.0 / a / a + 1.0 / b / b
end

function r8_uniform_01_device(seed::Int32)
    k = seed ÷ Int32(127773)
    seed = Int32(16807) * (seed - k * Int32(127773)) - k * Int32(2836)
    if seed < 0
        seed += Int32(2147483647)
    end
    return seed, Float64(seed) * 4.656612875e-10
end

function fk_kernel!(ni::Int32, nj::Int32, seed0::Int32, n_paths::Int32, a::Float64, b::Float64,
                    h::Float64, rth::Float64, n_inside, err)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= ni && j <= nj
        x = (Float64(nj - j) * (-a) + Float64(j - Int32(1)) * a) / Float64(nj - Int32(1))
        y = (Float64(ni - i) * (-b) + Float64(i - Int32(1)) * b) / Float64(ni - Int32(1))
        chk = (x / a)^2.0 + (y / b)^2.0

        if !(1.0 < chk)
            CUDA.@atomic n_inside[1] += Int32(1)
            w_exact = exp((x / a)^2.0 + (y / b)^2.0 - 1.0)
            wt = 0.0
            seed = seed0
            for _ in Int32(1):n_paths
                x1 = x
                x2 = y
                w = 1.0
                chk = 0.0
                while chk < 1.0
                    seed, ut = r8_uniform_01_device(seed)
                    if ut < 0.5
                        seed, us = r8_uniform_01_device(seed)
                        dx = us - 0.5 < 0.0 ? -rth : rth
                    else
                        dx = 0.0
                    end

                    seed, ut = r8_uniform_01_device(seed)
                    if ut < 0.5
                        seed, us = r8_uniform_01_device(seed)
                        dy = us - 0.5 < 0.0 ? -rth : rth
                    else
                        dy = 0.0
                    end

                    vs = potential_device(a, b, x1, x2)
                    x1 += dx
                    x2 += dy
                    vh = potential_device(a, b, x1, x2)
                    we = (1.0 - h * vs) * w
                    w = w - 0.5 * h * (vh * we + vs * w)
                    chk = (x1 / a)^2.0 + (x2 / b)^2.0
                end
                wt += w
            end
            wt /= Float64(n_paths)
            CUDA.@atomic err[1] += (w_exact - wt)^2.0
        end
    end
    return
end

function main()
    length(ARGS) == 1 || error("Usage: julia main.jl <iterations>")
    repeat = parse(Int, ARGS[1])
    a = 2.0
    b = 1.0
    dim = 2
    h = 0.001
    n_paths = Int32(1000)
    seed = Int32(123456789)

    println()
    println()
    println("FEYNMAN_KAC_2D:")
    println()
    println("  Program parameters:")
    println()
    println("  The calculation takes place inside a 2D ellipse.")
    println("  A rectangular grid of points will be defined.")
    println("  The solution will be estimated for those grid points")
    println("  from the point to the boundary.")
    println("  that lie inside the ellipse.")
    println()
    @printf("  Each solution will be estimated by computing %d trajectories\n", n_paths)
    println()
    println("    (X/A)^2 + (Y/B)^2 = 1")
    println()
    println("  The ellipse parameters A, B are set to:")
    println()
    @printf("    A = %f\n", a)
    @printf("    B = %f\n", b)
    @printf("  Stepsize H = %6.4f\n", h)

    rth = sqrt(Float64(dim) * h)
    nj = Int32(128)
    ni = Int32(1) + i4_ceiling(a / b) * (nj - Int32(1))
    println()
    @printf("  X coordinate marked by %d points\n", ni)
    @printf("  Y coordinate marked by %d points\n", nj)

    d_err = CUDA.zeros(Float64, 1)
    d_n_inside = CUDA.zeros(Int32, 1)
    threads = (16, 16)
    blocks = (cld(Int(ni), 16), cld(Int(nj), 16))
    total_ns = 0
    for _ in 1:repeat
        fill!(d_err, 0.0)
        fill!(d_n_inside, Int32(0))
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=threads blocks=blocks fk_kernel!(ni, nj, seed, n_paths, a, b, h, rth, d_n_inside, d_err)
        CUDA.synchronize()
        total_ns += time_ns() - t0
    end
    @printf("Average kernel time: %lf (s)\n", total_ns * 1e-9 / repeat)

    err = sqrt(Array(d_err)[1] / Float64(Array(d_n_inside)[1]))
    println()
    @printf("  RMS absolute error in solution = %e\n", err)
    println()
    println("FEYNMAN_KAC_2D:")
    println("  Normal end of execution.")
    println()
    println(err < 0.05 ? "PASS" : "FAIL")
end

main()
