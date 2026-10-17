using CUDA
using Printf
using Random

function mrc_gradient_kernel!(n::Int32, y, x1, x2, dout, margin::Float32, dx1, dx2)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            yf = Float32(y[i])
            dist = -yf * (x1[i] - x2[i]) + margin
            if dist < 0.0f0
                dx1[i] = 0.0f0
                dx2[i] = 0.0f0
            else
                dx1[i] = -yf * dout[i]
                dx2[i] = yf * dout[i]
            end
        end
    end
    return
end

function mrc_gradient2_kernel!(n::Int32, y, x1, x2, dout, margin::Float32, dx1, dx2)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            yf = Float32(y[i])
            o = dout[i]
            dist = -yf * (x1[i] - x2[i]) + margin
            dx1[i] = dist < 0.0f0 ? 0.0f0 : -yf * o
            dx2[i] = dist < 0.0f0 ? 0.0f0 : yf * o
        end
    end
    return
end

function mrc_gradient3_kernel!(n::Int32, y, x1, x2, dout, margin::Float32, dx1, dx2)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds begin
            yf = Float32(y[i])
            o = dout[i]
            dist = -yf * (x1[i] - x2[i]) + margin
            dx1[i] = dist < 0.0f0 ? 0.0f0 : -yf * o
            dx2[i] = dist < 0.0f0 ? 0.0f0 : yf * o
        end
        i += stride
    end
    return
end

function reference!(n::Int, y, x1, x2, dout, margin::Float32, dx1, dx2)
    for i in 1:n
        yf = Float32(y[i])
        dist = -yf * (x1[i] - x2[i]) + margin
        if dist < 0.0f0
            dx1[i] = 0.0f0
            dx2[i] = 0.0f0
        else
            dx1[i] = -yf * dout[i]
            dx2[i] = yf * dout[i]
        end
    end
end

function run_timed(label::String, kernel, repeat::Int, blocks::Int, n::Int,
                   y, x1, x2, dout, margin::Float32, dx1, dx2)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=256 blocks=blocks kernel(Int32(n), y, x1, x2, dout, margin, dx1, dx2)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - t0) * 1.0e-3 / repeat
    @printf("Average execution time of %s kernel: %f (us)\n", label, elapsed_us)
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <number of elements> <repeat>")
        return 1
    end

    length_n = parse(Int, args[1])
    repeat = parse(Int, args[2])
    margin = 0.01f0
    rng = MersenneTwister(123)

    h_x1 = rand(rng, Float32, length_n) .* 4.0f0 .- 2.0f0
    h_x2 = rand(rng, Float32, length_n) .* 4.0f0 .- 2.0f0
    h_o = rand(rng, Float32, length_n) .* 4.0f0 .- 2.0f0
    h_y = Vector{Int32}(undef, length_n)
    for i in 1:length_n
        h_y[i] = (rand(rng, Float32) * 4.0f0 - 2.0f0) < 0.0f0 ? Int32(-1) : Int32(1)
    end

    d_x1 = CuArray(h_x1)
    d_x2 = CuArray(h_x2)
    d_o = CuArray(h_o)
    d_y = CuArray(h_y)
    d_dx1 = CUDA.zeros(Float32, length_n)
    d_dx2 = CUDA.zeros(Float32, length_n)
    blocks = cld(length_n, 256)
    blocks3 = cld(max(length_n ÷ 4, 1), 256)

    for _ in 1:repeat
        @cuda threads=256 blocks=blocks mrc_gradient_kernel!(Int32(length_n), d_y, d_x1, d_x2, d_o, margin, d_dx1, d_dx2)
        @cuda threads=256 blocks=blocks mrc_gradient2_kernel!(Int32(length_n), d_y, d_x1, d_x2, d_o, margin, d_dx1, d_dx2)
        @cuda threads=256 blocks=blocks3 mrc_gradient3_kernel!(Int32(length_n), d_y, d_x1, d_x2, d_o, margin, d_dx1, d_dx2)
    end

    run_timed("MRC", mrc_gradient_kernel!, repeat, blocks, length_n, d_y, d_x1, d_x2, d_o, margin, d_dx1, d_dx2)
    run_timed("MRC2", mrc_gradient2_kernel!, repeat, blocks, length_n, d_y, d_x1, d_x2, d_o, margin, d_dx1, d_dx2)
    run_timed("MRC3", mrc_gradient3_kernel!, repeat, blocks3, length_n, d_y, d_x1, d_x2, d_o, margin, d_dx1, d_dx2)

    h_dx1 = Array(d_dx1)
    h_dx2 = Array(d_dx2)
    r_dx1 = Vector{Float32}(undef, length_n)
    r_dx2 = Vector{Float32}(undef, length_n)
    reference!(length_n, h_y, h_x1, h_x2, h_o, margin, r_dx1, r_dx2)

    ok = all(abs.(h_dx1 .- r_dx1) .<= 1.0f-3) && all(abs.(h_dx2 .- r_dx2) .<= 1.0f-3)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
