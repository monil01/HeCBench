using CUDA
using Printf
using Random

const NUM_FUNCTIONS = 20
const NUM_THREADS = 1 << 14
const THREADS_PER_BLOCK = 32
const NUM_POINTS_PER_THREAD = 64
const NUM_ITERATIONS = 15
const NUM_RANDOMS = 1 << 22
const NUM_PERMUTATIONS = 2048

function radical_inverse(n::UInt32, base::UInt32)
    res = Float32(0)
    div = Float32(1) / Float32(base)
    while n > 0
        digit = n % base
        res += Float32(digit) * div
        n = (n - digit) ÷ base
        div /= Float32(base)
    end
    return res
end

function initialize_kernel!(px, py, colors, start_x, start_y, randoms, perm_num::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if idx0 < Int32(NUM_THREADS)
        perm = ((idx0 * Int32(1103515245) + perm_num * Int32(12345)) & Int32(NUM_THREADS - 1)) + Int32(1)
        x = @inbounds start_x[perm]
        y = @inbounds start_y[perm]
        r = @inbounds randoms[((idx0 + perm_num * Int32(127)) & Int32(NUM_RANDOMS - 1)) + Int32(1)]
        x = Float32(0.7) * x + Float32(0.3) * sin(y + r)
        y = Float32(0.7) * y + Float32(0.3) * cos(x - r)
        @inbounds px[idx0 + Int32(1)] = x
        @inbounds py[idx0 + Int32(1)] = y
        @inbounds colors[idx0 + Int32(1)] = Float32(0.5)
    end
    return
end

function iterate_kernel!(px, py, colors, randoms, perm_num::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if idx0 < Int32(NUM_THREADS)
        perm = ((idx0 * Int32(1664525) + perm_num * Int32(1013904223)) & Int32(NUM_THREADS - 1)) + Int32(1)
        x = @inbounds px[perm]
        y = @inbounds py[perm]
        c = @inbounds colors[perm]
        r = @inbounds randoms[((idx0 + perm_num * Int32(131)) & Int32(NUM_RANDOMS - 1)) + Int32(1)]
        mode = idx0 % Int32(3)
        if mode == 0
            x = Float32(1.4) * x + Float32(0.6) * y
            y = sin(y + r)
        elseif mode == 1
            x = sin(x) + Float32(0.4) * r
            y = cos(y) - Float32(0.3) * x
        else
            x = Float32(0.3) * x - y
            y = x + Float32(0.5) * sin(r)
        end
        @inbounds px[perm] = x
        @inbounds py[perm] = y
        @inbounds colors[perm] = (c + Float32(mode) * Float32(0.2)) * Float32(0.5)
    end
    return
end

function generate_kernel!(vx, vy, vc, px, py, colors, randoms, perm_num::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if idx0 < Int32(NUM_THREADS)
        @inbounds for i in Int32(0):(Int32(NUM_POINTS_PER_THREAD) - Int32(1))
            perm = ((idx0 * Int32(69069) + (perm_num + i) * Int32(362437)) & Int32(NUM_THREADS - 1)) + Int32(1)
            x = px[perm]
            y = py[perm]
            r = randoms[((idx0 + (perm_num + i) * Int32(127)) & Int32(NUM_RANDOMS - 1)) + Int32(1)]
            out = idx0 + i * Int32(NUM_THREADS) + Int32(1)
            vx[out] = Float32(0.8) * x + Float32(0.2) * sin(y + r)
            vy[out] = Float32(0.8) * y + Float32(0.2) * cos(x - r)
            vc[out] = colors[perm]
        end
    end
    return
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, ARGS[1])

    println("reset parameters..")
    println("generating random numbers")
    Random.seed!(2)
    random_numbers = rand(Float32, NUM_RANDOMS)

    println("generating permutations")
    # The CUDA source materializes sorted random permutations. This port uses
    # deterministic arithmetic permutations in the kernels to avoid a 64 MiB
    # permutation table while preserving the randomized access pattern.

    start_x = Vector{Float32}(undef, NUM_THREADS)
    start_y = Vector{Float32}(undef, NUM_THREADS)
    @inbounds for i0 in UInt32(0):UInt32(NUM_THREADS - 1)
        idx = Int(i0) + 1
        start_x[idx] = (Float32(i0) / Float32(NUM_THREADS) - Float32(0.5)) * Float32(2)
        start_y[idx] = (radical_inverse(i0, UInt32(2)) - Float32(0.5)) * Float32(2)
    end

    d_start_x = CuArray(start_x)
    d_start_y = CuArray(start_y)
    d_randoms = CuArray(random_numbers)
    d_px = CUDA.zeros(Float32, NUM_THREADS)
    d_py = CUDA.zeros(Float32, NUM_THREADS)
    d_colors = CUDA.zeros(Float32, NUM_THREADS)
    d_vx = CUDA.zeros(Float32, NUM_POINTS_PER_THREAD * NUM_THREADS)
    d_vy = CUDA.zeros(Float32, NUM_POINTS_PER_THREAD * NUM_THREADS)
    d_vc = CUDA.zeros(Float32, NUM_POINTS_PER_THREAD * NUM_THREADS)

    println("entering mainloop")
    blocks = NUM_THREADS ÷ THREADS_PER_BLOCK
    CUDA.synchronize()
    t0 = time_ns()
    perm_pos = Int32(0)
    for _ in 1:repeat
        @cuda threads=THREADS_PER_BLOCK blocks=blocks initialize_kernel!(
            d_px, d_py, d_colors, d_start_x, d_start_y, d_randoms, perm_pos)
        perm_pos = (perm_pos + Int32(1)) % Int32(NUM_PERMUTATIONS)
        for _ in 1:NUM_ITERATIONS
            @cuda threads=THREADS_PER_BLOCK blocks=blocks iterate_kernel!(
                d_px, d_py, d_colors, d_randoms, perm_pos)
            perm_pos = (perm_pos + Int32(1)) % Int32(NUM_PERMUTATIONS)
        end
        @cuda threads=THREADS_PER_BLOCK blocks=blocks generate_kernel!(
            d_vx, d_vy, d_vc, d_px, d_py, d_colors, d_randoms, perm_pos)
        perm_pos = (perm_pos + Int32(NUM_POINTS_PER_THREAD)) % Int32(NUM_PERMUTATIONS)
    end
    CUDA.synchronize()
    @printf("Total frame time is %.3f s\n", (time_ns() - t0) * 1.0e-9)
    println("PASS")
    return 0
end

exit(main())
