using CUDA
using Printf
using Random

@inline function round_nearest_i32(x::Float64)
    return round(Int32, x)
end

@inline function zero_point_one(
    min_in::Float32,
    max_in::Float32,
    qmin::Int32,
    qmax::Int32,
    preserve_sparsity::Bool,
)
    min_val = min_in
    max_val = max_in

    if min_val < 0.0f0 && max_val > 0.0f0 && preserve_sparsity
        symmetric_qmin = -((qmax - qmin) ÷ Int32(2) + Int32(1))
        symmetric_qmax = (qmax - qmin) ÷ Int32(2)
        max_scale = max(abs(Float64(min_val) / Float64(symmetric_qmin)),
                        abs(Float64(max_val) / Float64(symmetric_qmax)))
        min_val = Float32(max_scale * Float64(symmetric_qmin))
        max_val = Float32(max_scale * Float64(symmetric_qmax))
    end

    min_val = min(min_val, 0.0f0)
    max_val = max(max_val, 0.0f0)
    scale = Float32((Float64(max_val) - Float64(min_val)) / Float64(qmax - qmin))

    if scale == 0.0f0 || isinf(1.0f0 / scale)
        scale = 0.1f0
    end

    zero_point_from_min = Float64(qmin) - Float64(min_val) / Float64(scale)
    zero_point_from_max = Float64(qmax) - Float64(max_val) / Float64(scale)
    zero_point_from_min_error = abs(Float64(qmin)) + abs(Float64(min_val) / Float64(scale))
    zero_point_from_max_error = abs(Float64(qmax)) + abs(Float64(max_val) / Float64(scale))
    initial_zero_point = zero_point_from_min_error < zero_point_from_max_error ?
                         zero_point_from_min : zero_point_from_max

    if min_val < 0.0f0 && max_val > 0.0f0 && preserve_sparsity
        initial_zero_point = Float64(qmin + qmax) / 2.0
    end

    nudged = if initial_zero_point < Float64(qmin)
        qmin
    elseif initial_zero_point > Float64(qmax)
        qmax
    else
        round_nearest_i32(initial_zero_point)
    end
    return scale, nudged
end

function zero_point_kernel!(x_min, x_max, qmin::Int32, qmax::Int32, n::Int32,
                            preserve_sparsity::Bool, scale, zp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds s, z = zero_point_one(x_min[i], x_max[i], qmin, qmax, preserve_sparsity)
        @inbounds scale[i] = s
        @inbounds zp[i] = z
    end
    return
end

function reference!(x_min, x_max, qmin, qmax, preserve_sparsity, scale, zp)
    for i in eachindex(x_min)
        scale[i], zp[i] = zero_point_one(x_min[i], x_max[i], qmin, qmax, preserve_sparsity)
    end
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <number of min/max values> <repeat>")
        return 1
    end

    n = parse(Int, args[1])
    repeat = parse(Int, args[2])
    qmin = Int32(-127)
    qmax = Int32(127)
    preserve_sparsity = true

    rng = MersenneTwister(123)
    x_min = rand(rng, Float32, n) .* 2.0f0 .- 1.0f0
    x_max = rand(rng, Float32, n) .* 2.0f0 .- 1.0f0
    scale_ref = Vector{Float32}(undef, n)
    zp_ref = Vector{Int32}(undef, n)
    reference!(x_min, x_max, qmin, qmax, preserve_sparsity, scale_ref, zp_ref)

    d_min = CuArray(x_min)
    d_max = CuArray(x_max)
    d_scale = CUDA.zeros(Float32, n)
    d_zp = CUDA.zeros(Int32, n)

    threads = 256
    blocks = cld(n, threads)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks zero_point_kernel!(
            d_min, d_max, qmin, qmax, Int32(n), preserve_sparsity, d_scale, d_zp)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of zero-point kernel: %f (us)\n", elapsed_us)

    scale = Array(d_scale)
    zp = Array(d_zp)
    ok = true
    for i in 1:n
        if zp[i] != zp_ref[i] || abs(scale[i] - scale_ref[i]) > 1.0f-3
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
