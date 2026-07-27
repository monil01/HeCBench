using CUDA
using Printf

# Julia port of haversine-cuda benchmark.
# Simplification: instead of a struct-of-4 layout (double4), we pack lat/lon
# arrays as four separate CuArray{Float64}. Math and result are identical.

const DEGREE_TO_RADIAN = pi / 180.0
const EARTH_RADIUS_KM  = 6371.0

function haversine_kernel!(ay_arr, ax_arr, by_arr, bx_arr, dist, N::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    @inbounds if i <= N
        deg2rad = 3.141592653589793 / 180.0
        ay = ay_arr[i] * deg2rad
        ax = ax_arr[i] * deg2rad
        by = by_arr[i] * deg2rad
        bx = bx_arr[i] * deg2rad
        x = (bx - ax) / 2.0
        y = (by - ay) / 2.0
        sinysqrd = sin(y) * sin(y)
        sinxsqrd = sin(x) * sin(x)
        scale = cos(ay) * cos(by)
        dist[i] = 2.0 * 6371.0 * asin(sqrt(sinysqrd + sinxsqrd * scale))
    end
    return
end

function main()
    if length(ARGS) < 2
        println("Usage: main.jl <locations.txt> <repeat>")
        return 1
    end
    filename = ARGS[1]
    repeat_n = parse(Int, ARGS[2])
    println("Reading city locations from file $filename...")

    num_cities = 2097152
    num_ref = 6
    index_map = [436483, 1952407, 627919, 377884, 442703, 1863423]  # 1-based? source is C 1-based -> matches file lines 1..N
    N = num_cities * num_ref

    lats = Vector{Float64}(undef, num_cities)
    lons = Vector{Float64}(undef, num_cities)
    open(filename, "r") do fp
        for i in 1:num_cities
            line = readline(fp)
            parts = split(line)
            lats[i] = parse(Float64, parts[1])
            lons[i] = parse(Float64, parts[2])
        end
    end

    # ay=lat_a, ax=lon_a, by=lat_b, bx=lon_b
    ay_h = Vector{Float64}(undef, N)
    ax_h = Vector{Float64}(undef, N)
    by_h = Vector{Float64}(undef, N)
    bx_h = Vector{Float64}(undef, N)
    for c in 0:(num_ref-1)
        idx = index_map[c+1]  # CUDA does index_map[c]-1 then uses 0-based; the 1-based file line is index_map[c]
        blat = lats[idx]
        blon = lons[idx]
        for j in 1:num_cities
            gid = c * num_cities + j
            ay_h[gid] = lats[j]
            ax_h[gid] = lons[j]
            by_h[gid] = blat
            bx_h[gid] = blon
        end
    end

    d_ay = CuArray(ay_h); d_ax = CuArray(ax_h)
    d_by = CuArray(by_h); d_bx = CuArray(bx_h)
    d_dist = CUDA.zeros(Float64, N)

    threads = 256
    blocks  = cld(N, threads)
    # Warmup
    @cuda threads=threads blocks=blocks haversine_kernel!(d_ay, d_ax, d_by, d_bx, d_dist, Int32(N))
    CUDA.synchronize()

    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks haversine_kernel!(d_ay, d_ax, d_by, d_bx, d_dist, Int32(N))
    end
    CUDA.synchronize()
    ktime_s = (time_ns() - t0) * 1e-9 / repeat_n
    @printf("Average kernel execution time %f (s)\n", ktime_s)

    out = Array(d_dist)

    # CPU reference
    maxerr = 0.0
    d2r = pi / 180.0
    for i in 1:N
        ax = ax_h[i] * d2r
        ay = ay_h[i] * d2r
        bx = bx_h[i] * d2r
        by = by_h[i] * d2r
        x = (bx - ax) / 2.0
        y = (by - ay) / 2.0
        sinysqrd = sin(y) * sin(y)
        sinxsqrd = sin(x) * sin(x)
        scale = cos(ay) * cos(by)
        ref = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale))
        e = abs(out[i] - ref)
        if e > maxerr; maxerr = e; end
    end
    @printf("The maximum error in distance is %f\n", maxerr)
    println(maxerr < 1e-6 ? "PASS" : "FAIL")
    return 0
end

main()
