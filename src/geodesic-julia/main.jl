using CUDA
using Printf

const GDC_DEG_TO_RAD = Float32(3.141592654 / 180.0)
const GDC_FLATTENING = Float32(1.0 - (6356752.31424518 / 6378137.0))
const GDC_ECCENTRICITY = Float32(6356752.31424518 / 6378137.0)
const GDC_ELLIPSOIDAL = Float32(1.0 / (6356752.31414 / 6378137.0)^2 - 1.0)
const GC_SEMI_MINOR = Float32(6356752.31424518)
const EPS = Float32(0.5e-5)

@inline function distance_one(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32)
    if lat1 == lat2 && lon1 == lon2
        return 0.0f0
    end
    rad_lon_1 = lon1 * GDC_DEG_TO_RAD
    rad_lat_1 = lat1 * GDC_DEG_TO_RAD
    rad_lon_2 = lon2 * GDC_DEG_TO_RAD
    rad_lat_2 = lat2 * GDC_DEG_TO_RAD

    tu1 = GDC_ECCENTRICITY * sin(rad_lat_1) / cos(rad_lat_1)
    tu2 = GDC_ECCENTRICITY * sin(rad_lat_2) / cos(rad_lat_2)
    cu1 = Float32(1) / sqrt(tu1 * tu1 + Float32(1))
    su1 = cu1 * tu1
    cu2 = Float32(1) / sqrt(tu2 * tu2 + Float32(1))
    dist = cu1 * cu2
    baz = dist * tu2
    faz = baz * tu1
    x = rad_lon_2 - rad_lon_1
    c2a = Float32(0)
    cy = Float32(0)
    cz = Float32(0)
    e = Float32(0)
    sy = Float32(0)
    y = Float32(0)
    iter = 0
    while true
        sx = sin(x)
        cx = cos(x)
        tu1 = cu2 * sx
        tu2 = baz - su1 * cu2 * cx
        sy = sqrt(tu1 * tu1 + tu2 * tu2)
        cy = dist * cx + faz
        y = atan(sy, cy)
        sa = dist * sx / sy
        c2a = -sa * sa + Float32(1)
        cz = faz + faz
        if c2a > 0.0f0
            cz = -cz / c2a + cy
        end
        e = cz * cz * Float32(2) - Float32(1)
        c = (((-Float32(3) * c2a + Float32(4)) * GDC_FLATTENING + Float32(4)) *
             c2a * GDC_FLATTENING / Float32(16))
        d = x
        x = ((e * cy * c + cz) * sy * c + y) * sa
        x = (Float32(1) - c) * x * GDC_FLATTENING + rad_lon_2 - rad_lon_1
        iter += 1
        (abs(d - x) <= EPS || iter >= 100) && break
    end
    x = sqrt(GDC_ELLIPSOIDAL * c2a + Float32(1)) + Float32(1)
    x = (x - Float32(2)) / x
    c = Float32(1) - x
    c = (x * x / Float32(4) + Float32(1)) / c
    d = (Float32(0.375) * x * x - Float32(1)) * x
    x = e * cy
    dist = Float32(1) - e - e
    return ((((sy * sy * Float32(4) - Float32(3)) * dist * cz * d / Float32(6) -
              x) * d / Float32(4) + cz) * sy * d + y) * c * GC_SEMI_MINOR
end

function distance_kernel!(lat1, lon1, lat2, lon2, out, n::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int32(blockDim().x * gridDim().x)
    while i <= n
        @inbounds out[i] = distance_one(lat1[i], lon1[i], lat2[i], lon2[i])
        i += stride
    end
    return
end

function read_locations(path::String)
    lats = Float32[]
    lons = Float32[]
    open(path, "r") do io
        for line in eachline(io)
            parts = split(strip(line))
            length(parts) >= 2 || continue
            push!(lats, parse(Float32, parts[1]))
            push!(lons, parse(Float32, parts[2]))
        end
    end
    return lats, lons
end

function main(args)
    if length(args) != 2
        println("Usage main.jl <path to city locations> <repeat>")
        return 1
    end
    filename = args[1]
    iteration = parse(Int, args[2])
    println("Reading city locations from file $filename...")
    lats, lons = read_locations(filename)
    isempty(lats) && error("no city locations read")

    ref_count = min(6, length(lats))
    ncity = length(lats)
    n = ncity * ref_count
    lat1 = Vector{Float32}(undef, n)
    lon1 = Vector{Float32}(undef, n)
    lat2 = Vector{Float32}(undef, n)
    lon2 = Vector{Float32}(undef, n)
    @inbounds for c in 1:ref_count
        ref = min(c, ncity)
        for j in 1:ncity
            idx = (c - 1) * ncity + j
            lat1[idx] = lats[j]
            lon1[idx] = lons[j]
            lat2[idx] = lats[ref]
            lon2[idx] = lons[ref]
        end
    end
    expected = [distance_one(lat1[i], lon1[i], lat2[i], lon2[i]) for i in 1:n]
    d_lat1, d_lon1 = CuArray(lat1), CuArray(lon1)
    d_lat2, d_lon2 = CuArray(lat2), CuArray(lon2)
    d_out = CUDA.zeros(Float32, n)
    threads = 256
    blocks = max(cld(n, threads), 1)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iteration
        @cuda threads=threads blocks=blocks distance_kernel!(d_lat1, d_lon1, d_lat2, d_lon2, d_out, Int32(n))
    end
    CUDA.synchronize()
    @printf("Average kernel execution time %f (us)\n", (time_ns() - t0) * 1e-3 / iteration)
    output = Array(d_out)
    @printf("The maximum error in distance for single precision is %f\n", maximum(abs.(output .- expected)))
    return 0
end

exit(main(ARGS))
