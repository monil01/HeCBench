using CUDA
using Printf
using Random

const DEGREE_TO_RADIAN_F64 = pi / 180.0
const EARTH_CIRCUMFERENCE_KM_PER_DEGREE_F64 = 40000.0 / 360.0

function transform_kernel!(lon, lat, outx, outy, n::Int32, ox, oy, degree_to_radian, km_per_degree)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end
    @inbounds begin
        mid = (lat[i] + oy) / typeof(oy)(2)
        outx[i] = (ox - lon[i]) * km_per_degree * cos(mid * degree_to_radian)
        outy[i] = (oy - lat[i]) * km_per_degree
    end
    return
end

function host_transform(lon::Vector{T}, lat::Vector{T}) where {T}
    ox = T(90)
    oy = T(45)
    degree_to_radian = DEGREE_TO_RADIAN_F64
    km_per_degree = EARTH_CIRCUMFERENCE_KM_PER_DEGREE_F64
    outx = similar(lon)
    outy = similar(lat)
    @inbounds for i in eachindex(lon)
        mid = (lat[i] + oy) / T(2)
        outx[i] = (ox - lon[i]) * km_per_degree * cos(mid * degree_to_radian)
        outy[i] = (oy - lat[i]) * km_per_degree
    end
    return outx, outy
end

function coordinates_transform(::Type{T}, num_coords::Int, repeat::Int) where {T}
    @printf("Number of coordinates is %d and coordinate size is %d bytes\n", num_coords, 2 * sizeof(T))
    rng = MersenneTwister(123)
    lon = T.(rand(rng, -180:179, num_coords))
    lat = T.(rand(rng, -90:89, num_coords))
    d_lon = CuArray(lon)
    d_lat = CuArray(lat)
    d_outx = CUDA.zeros(T, num_coords)
    d_outy = CUDA.zeros(T, num_coords)

    threads = 256
    blocks = cld(num_coords, threads)
    ox = T(90)
    oy = T(45)
    degree_to_radian = DEGREE_TO_RADIAN_F64
    km_per_degree = EARTH_CIRCUMFERENCE_KM_PER_DEGREE_F64

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks transform_kernel!(
            d_lon, d_lat, d_outx, d_outy, Int32(num_coords), ox, oy,
            degree_to_radian, km_per_degree)
    end
    CUDA.synchronize()
    elapsed_us = (time_ns() - start) * 1.0e-3 / repeat
    @printf("Average execution time of device transform: %f (us)\n", elapsed_us)

    outx = Array(d_outx)
    outy = Array(d_outy)

    start = time_ns()
    refx = similar(lon)
    refy = similar(lat)
    for _ in 1:10
        refx, refy = host_transform(lon, lat)
    end
    elapsed_host = (time_ns() - start) * 1.0e-3 / 10
    @printf("Average execution time of host transform: %f (us)\n", elapsed_host)

    ok = all(abs.(outx .- refx) .< T(1.0e-3)) && all(abs.(outy .- refy) .< T(1.0e-3))
    println(ok ? "PASS" : "FAIL")
    return ok
end

function main(args)
    if length(args) != 2
        println("Usage: ./main <number of coordinates> <repeat>")
        return 1
    end
    num_coords = parse(Int, args[1])
    repeat = parse(Int, args[2])

    println()
    println("Double-precision coordinates transform")
    ok64 = coordinates_transform(Float64, num_coords, repeat)

    println()
    println("Single-precision coordinates transform")
    ok32 = coordinates_transform(Float32, num_coords, repeat)

    return ok64 && ok32 ? 0 : 1
end

exit(main(ARGS))
