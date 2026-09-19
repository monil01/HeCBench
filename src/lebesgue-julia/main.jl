using CUDA
using Printf
using Dates

const R8_PI = 3.141592653589793

function lebesgue_kernel!(vals, xfun, x, n::Int32, nfun::Int32)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if j > nfun
        return
    end
    t = 0.0
    @inbounds for i1 in Int32(1):n
        interp = 1.0
        for i2 in Int32(1):n
            if i1 != i2
                interp *= (xfun[j] - x[i2]) / (x[i1] - x[i2])
            end
        end
        t += abs(interp)
    end
    @inbounds vals[j] = t
    return
end

function linspace_vec(n::Int, a::Float64, b::Float64)
    n == 1 && return [(a + b) / 2.0]
    [((n - 1 - i) * a + i * b) / (n - 1) for i in 0:(n - 1)]
end

chebyshev1(n) = [cos(R8_PI * (2 * i + 1) / (2 * n)) for i in 0:(n - 1)]

function chebyshev2(n)
    n == 1 && return [0.0]
    [cos(R8_PI * (n - i - 1) / (n - 1)) for i in 0:(n - 1)]
end

chebyshev3(n) = [cos(R8_PI * (2 * n - 2 * i - 1) / (2 * n + 1)) for i in 0:(n - 1)]
chebyshev4(n) = [cos(R8_PI * (2 * n - 2 * i) / (2 * n + 1)) for i in 0:(n - 1)]
equidistant1(n) = [(-n + 1 + 2 * i) / (n + 1) for i in 0:(n - 1)]

function equidistant2(n)
    n == 1 && return [0.0]
    [(-n + 1 + 2 * i) / (n - 1) for i in 0:(n - 1)]
end

equidistant3(n) = [(-n + 1 + 2 * i) / n for i in 0:(n - 1)]
fejer1(n) = [cos(R8_PI * (2 * n - 1 - 2 * i) / (2 * n)) for i in 0:(n - 1)]
fejer2(n) = [cos(R8_PI * (n - i) / (n + 1)) for i in 0:(n - 1)]

function lebesgue_constant(x::Vector{Float64}, d_xfun, nfun::Int)
    n = length(x)
    n == 1 && return 1.0
    d_vals = CUDA.zeros(Float64, nfun)
    d_x = CuArray(x)
    @cuda threads=256 blocks=cld(nfun, 256) lebesgue_kernel!(d_vals, d_xfun, d_x, Int32(n), Int32(nfun))
    CUDA.synchronize()
    return maximum(d_vals)
end

function run_test(id::Int, title::String, generator, nfun::Int)
    println()
    @printf("LEBESGUE_TEST%02d:\n", id)
    println("  Analyze $title points.")

    xfun = linspace_vec(nfun, -1.0, 1.0)
    d_xfun = CuArray(xfun)
    total_time = 0.0
    ok = true
    for n in 1:11
        x = generator(n)
        CUDA.synchronize()
        start = time_ns()
        l = lebesgue_constant(x, d_xfun, nfun)
        CUDA.synchronize()
        total_time += time_ns() - start
        ok &= isfinite(l) && l >= 1.0 && (n != 1 || abs(l - 1.0) <= 1.0e-12)
    end
    @printf("  Total kernel execution time %f (s)\n", total_time * 1.0e-9)
    @printf("  %s\n", ok ? "PASS" : "FAIL")
    return ok
end

function timestamp()
    println(Dates.format(now(), "dd u yyyy HH:MM:SS"))
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <number of points in an interval> <repeat>")
        return 1
    end
    nfun = parse(Int, args[1])
    repeat = parse(Int, args[2])

    println()
    println("LEBESGUE_TEST")
    ok = true
    tests = [
        (1, "Chebyshev1", chebyshev1),
        (2, "Chebyshev2", chebyshev2),
        (3, "Chebyshev3", chebyshev3),
        (4, "Chebyshev4", chebyshev4),
        (5, "Equidistant1", equidistant1),
        (6, "Equidistant2", equidistant2),
        (7, "Equidistant3", equidistant3),
        (8, "Fejer1", fejer1),
        (9, "Fejer2", fejer2),
    ]
    for _ in 1:repeat
        timestamp()
        for (id, title, gen) in tests
            ok &= run_test(id, title, gen, nfun)
        end
    end
    return ok ? 0 : 1
end

exit(main(ARGS))
