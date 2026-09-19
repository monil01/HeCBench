using CUDA
using Printf
using Random

function generate_rects(size::Float32)
    phi = Float32((1 + sqrt(5.0)) * 0.5)
    tlx = 0.0f0
    tly = 0.0f0
    brx = size
    bry = size
    rects = Tuple{Int,Float32,Float32,Float32,Float32}[]
    total_points = 0
    while true
        k = length(rects) % 4
        if k == 0
            brx = tlx - (tlx - brx) / phi
        elseif k == 1
            bry = tly - (tly - bry) / phi
        elseif k == 2
            tlx = tlx + (brx - tlx) / phi
        else
            tly = tly + (bry - tly) / phi
        end
        area_x = brx - tlx
        area_y = bry - tly
        n = Int(floor(sqrt(Float64(area_x * area_y * 1_000_000.0f0))))
        push!(rects, (n, tlx, tly, brx, bry))
        total_points += n
        if !(area_x > 1.0f0 && area_y > 1.0f0)
            break
        end
    end
    return total_points, rects
end

function generate_points(size::Float32)
    total_points, rects = generate_rects(size)
    rng = MersenneTwister(123)
    x = Vector{Float32}(undef, total_points)
    y = Vector{Float32}(undef, total_points)
    offset = 0
    for (n, tlx, tly, brx, bry) in rects
        @inbounds for j in 1:n
            x[offset + j] = tlx + (brx - tlx) * rand(rng, Float32)
            y[offset + j] = tly + (bry - tly) * rand(rng, Float32)
        end
        offset += n
    end
    return x, y
end

function eval_minmax(bounding_box_size::Float32, repeat::Int)
    x, y = generate_points(bounding_box_size)
    total_points = length(x)
    @printf("Total number of points: %d\n", total_points)

    d_x = CuArray(x)
    d_y = CuArray(y)
    d_mag = CUDA.zeros(Float32, total_points)
    min_pair = (0.0f0, 0.0f0)
    max_pair = (0.0f0, 0.0f0)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        d_mag .= d_x .* d_x .+ d_y .* d_y
        imin = argmin(d_mag)
        imax = argmax(d_mag)
        min_pair = (x[imin], y[imin])
        max_pair = (x[imax], y[imax])
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of thrust:min() and thrust:max(): %f (us)\n", elapsed_us)

    minmax_min_pair = min_pair
    minmax_max_pair = max_pair
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        d_mag .= d_x .* d_x .+ d_y .* d_y
        imin = argmin(d_mag)
        imax = argmax(d_mag)
        minmax_min_pair = (x[imin], y[imin])
        minmax_max_pair = (x[imax], y[imax])
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of thrust:min_max(): %f (us)\n", elapsed_us)

    mags = x .* x .+ y .* y
    imin = argmin(mags)
    imax = argmax(mags)
    ref_min = (x[imin], y[imin])
    ref_max = (x[imax], y[imax])
    ok = min_pair == ref_min && max_pair == ref_max &&
         minmax_min_pair == ref_min && minmax_max_pair == ref_max
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <bounding-box size> <repeat>")
        return 1
    end
    size = parse(Int, args[1])
    repeat = parse(Int, args[2])
    ok = eval_minmax(Float32(size), repeat)
    return ok ? 0 : 1
end

exit(main(ARGS))
