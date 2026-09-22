using CUDA
using Printf
using Random

const WIDTH = 256
const HEIGHT = 256
const OMEGA = 1.2
const DIRX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const DIRY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const WEIGHT = (4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0)

@inline function ced(rho, weight, dx, dy, ux, uy)
    u2 = ux * ux + uy * uy
    eu = dx * ux + dy * uy
    return rho * weight * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
end

function initialize()
    n = WIDTH * HEIGHT
    if0 = Vector{Float64}(undef, n)
    if1234 = Vector{Float64}(undef, n * 4)
    if5678 = Vector{Float64}(undef, n * 4)
    types = Vector{UInt8}(undef, n)
    rng = MersenneTwister(123)
    ux = 0.01
    uy = 0.01
    @inbounds for y in 0:(HEIGHT - 1), x in 0:(WIDTH - 1)
        pos = x + y * WIDTH + 1
        den = rand(rng, 1:10)
        if0[pos] = ced(den, WEIGHT[1], DIRX[1], DIRY[1], ux, uy)
        for k in 1:4
            if1234[(pos - 1) * 4 + k] = ced(den, WEIGHT[k + 1], DIRX[k + 1], DIRY[k + 1], ux, uy)
            if5678[(pos - 1) * 4 + k] = ced(den, WEIGHT[k + 5], DIRX[k + 5], DIRY[k + 5], ux, uy)
        end
        types[pos] = (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) ? UInt8(1) : UInt8(0)
    end
    return if0, if1234, if5678, types
end

function lbm_kernel!(if0, of0, if1234, of1234, if5678, of5678, types, weights, omega::Float64)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    width = Int32(WIDTH)
    height = Int32(HEIGHT)
    pos0 = x + width * y
    pos = pos0 + Int32(1)

    f0 = @inbounds if0[pos]
    f1 = @inbounds if1234[pos0 * Int32(4) + Int32(1)]
    f2 = @inbounds if1234[pos0 * Int32(4) + Int32(2)]
    f3 = @inbounds if1234[pos0 * Int32(4) + Int32(3)]
    f4 = @inbounds if1234[pos0 * Int32(4) + Int32(4)]
    f5 = @inbounds if5678[pos0 * Int32(4) + Int32(1)]
    f6 = @inbounds if5678[pos0 * Int32(4) + Int32(2)]
    f7 = @inbounds if5678[pos0 * Int32(4) + Int32(3)]
    f8 = @inbounds if5678[pos0 * Int32(4) + Int32(4)]

    e0 = f0
    e1 = f1; e2 = f2; e3 = f3; e4 = f4
    e5 = f5; e6 = f6; e7 = f7; e8 = f8
    if @inbounds(types[pos]) == UInt8(1)
        e1 = f3; e2 = f4; e3 = f1; e4 = f2
        e5 = f7; e6 = f8; e7 = f5; e8 = f6
    else
        rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
        uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho
        e0 = (1.0 - omega) * f0 + omega * ced(rho, weights[1], 0.0, 0.0, ux, uy)
        e1 = (1.0 - omega) * f1 + omega * ced(rho, weights[2], 1.0, 0.0, ux, uy)
        e2 = (1.0 - omega) * f2 + omega * ced(rho, weights[3], 0.0, 1.0, ux, uy)
        e3 = (1.0 - omega) * f3 + omega * ced(rho, weights[4], -1.0, 0.0, ux, uy)
        e4 = (1.0 - omega) * f4 + omega * ced(rho, weights[5], 0.0, -1.0, ux, uy)
        e5 = (1.0 - omega) * f5 + omega * ced(rho, weights[6], 1.0, 1.0, ux, uy)
        e6 = (1.0 - omega) * f6 + omega * ced(rho, weights[7], -1.0, 1.0, ux, uy)
        e7 = (1.0 - omega) * f7 + omega * ced(rho, weights[8], -1.0, -1.0, ux, uy)
        e8 = (1.0 - omega) * f8 + omega * ced(rho, weights[9], 1.0, -1.0, ux, uy)
    end

    if x > 0 && x < width - Int32(1) && y > 0 && y < height - Int32(1)
        @inbounds of0[pos] = e0
        @inbounds of1234[(pos0 + Int32(1)) * Int32(4) + Int32(1)] = e1
        @inbounds of1234[(pos0 + width) * Int32(4) + Int32(2)] = e2
        @inbounds of1234[(pos0 - Int32(1)) * Int32(4) + Int32(3)] = e3
        @inbounds of1234[(pos0 - width) * Int32(4) + Int32(4)] = e4
        @inbounds of5678[(pos0 + width + Int32(1)) * Int32(4) + Int32(1)] = e5
        @inbounds of5678[(pos0 + width - Int32(1)) * Int32(4) + Int32(2)] = e6
        @inbounds of5678[(pos0 - width - Int32(1)) * Int32(4) + Int32(3)] = e7
        @inbounds of5678[(pos0 - width + Int32(1)) * Int32(4) + Int32(4)] = e8
    end
    return
end

function simulate_gpu(iterations, if0, if1234, if5678, types)
    d_if0 = CuArray(if0); d_of0 = CuArray(if0)
    d_if1234 = CuArray(if1234); d_of1234 = CuArray(if1234)
    d_if5678 = CuArray(if5678); d_of5678 = CuArray(if5678)
    d_types = CuArray(types)
    d_weights = CuArray(collect(Float64, WEIGHT))
    threads = (256, 1)
    blocks = (1, HEIGHT)
    @cuda threads=threads blocks=blocks lbm_kernel!(d_if0, d_of0, d_if1234, d_of1234, d_if5678, d_of5678, d_types, d_weights, OMEGA)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks lbm_kernel!(d_if0, d_of0, d_if1234, d_of1234, d_if5678, d_of5678, d_types, d_weights, OMEGA)
        d_if0, d_of0 = d_of0, d_if0
        d_if1234, d_of1234 = d_of1234, d_if1234
        d_if5678, d_of5678 = d_of5678, d_if5678
    end
    CUDA.synchronize()
    elapsed = time_ns() - t0
    return Array(d_if0), Array(d_if1234), Array(d_if5678), elapsed
end

function collide_reference!(ef0, ef1234, ef5678, if0, if1234, if5678, types)
    @inbounds for y in 0:(HEIGHT - 1), x in 0:(WIDTH - 1)
        p0 = x + WIDTH * y
        p = p0 + 1
        f0 = if0[p]
        f1, f2, f3, f4 = if1234[p0*4+1], if1234[p0*4+2], if1234[p0*4+3], if1234[p0*4+4]
        f5, f6, f7, f8 = if5678[p0*4+1], if5678[p0*4+2], if5678[p0*4+3], if5678[p0*4+4]
        if types[p] == UInt8(1)
            ef0[p] = f0
            ef1234[p0*4+1] = f3; ef1234[p0*4+2] = f4; ef1234[p0*4+3] = f1; ef1234[p0*4+4] = f2
            ef5678[p0*4+1] = f7; ef5678[p0*4+2] = f8; ef5678[p0*4+3] = f5; ef5678[p0*4+4] = f6
        else
            rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
            uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho
            fs = (f0, f1, f2, f3, f4, f5, f6, f7, f8)
            vals = ntuple(k -> (1.0 - OMEGA) * fs[k] + OMEGA * ced(rho, WEIGHT[k], DIRX[k], DIRY[k], ux, uy), 9)
            ef0[p] = vals[1]
            for k in 1:4
                ef1234[p0*4+k] = vals[k+1]
                ef5678[p0*4+k] = vals[k+5]
            end
        end
    end
end

function reference(iterations, if0, if1234, if5678, types)
    vf0 = copy(if0); vf1234 = copy(if1234); vf5678 = copy(if5678)
    of0 = copy(if0); of1234 = copy(if1234); of5678 = copy(if5678)
    ef0 = similar(if0); ef1234 = similar(if1234); ef5678 = similar(if5678)
    @inbounds for _ in 1:iterations
        collide_reference!(ef0, ef1234, ef5678, vf0, vf1234, vf5678, types)
        for y in 1:(HEIGHT - 2), x in 1:(WIDTH - 2)
            src = x + WIDTH * y
            of0[src + 1] = ef0[src + 1]
            for k in 1:9
                dst = x + DIRX[k] + WIDTH * (y + DIRY[k])
                if k >= 2 && k <= 5
                    of1234[dst * 4 + k - 1] = ef1234[src * 4 + k - 1]
                elseif k >= 6
                    of5678[dst * 4 + k - 5] = ef5678[src * 4 + k - 5]
                end
            end
        end
        vf0, of0 = of0, vf0
        vf1234, of1234 = of1234, vf1234
        vf5678, of5678 = of5678, vf5678
    end
    return vf0, vf1234, vf5678
end

function verify(g0, g1234, g5678, r0, r1234, r5678)
    ok = maximum(abs.(g0 .- r0)) <= 1.0e-3 &&
         maximum(abs.(g1234 .- r1234)) <= 1.0e-3 &&
         maximum(abs.(g5678 .- r5678)) <= 1.0e-3
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 1
        println("Usage main.jl <iterations>")
        return 1
    end
    iterations = parse(Int, args[1])
    if0, if1234, if5678, types = initialize()
    r0, r1234, r5678 = reference(iterations, if0, if1234, if5678, types)
    g0, g1234, g5678, elapsed = simulate_gpu(iterations, if0, if1234, if5678, types)
    @printf("Average kernel execution time %f (s)\n", elapsed * 1.0e-9 / iterations)
    return verify(g0, g1234, g5678, r0, r1234, r5678) ? 0 : 1
end

exit(main(ARGS))
