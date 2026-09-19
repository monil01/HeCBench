using CUDA
using Printf
using Random

const NUM_ELEMENTS = 10_000_000
const BLOCK_SIZE = 256
const K_PI_VALUE = 3.1415f0
const K_G_VALUE = 9.81f0

function calculate_range_kernel!(angle, velocity, range, total_time, max_height, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end
    @inbounds begin
        proj_angle = angle[i]
        proj_vel = velocity[i]
        sin_value = sin(proj_angle * K_PI_VALUE / 180.0f0)
        cos_value = cos(proj_angle * K_PI_VALUE / 180.0f0)
        t = abs((2.0f0 * proj_vel * sin_value)) / K_G_VALUE
        r = abs(proj_vel * t * cos_value)
        h = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f0 * K_G_VALUE
        range[i] = r
        total_time[i] = t
        max_height[i] = h
    end
    return
end

function reference(angle, velocity)
    range = similar(angle)
    total_time = similar(angle)
    max_height = similar(angle)
    @inbounds for i in eachindex(angle)
        proj_angle = angle[i]
        proj_vel = velocity[i]
        sin_value = sin(proj_angle * K_PI_VALUE / 180.0f0)
        cos_value = cos(proj_angle * K_PI_VALUE / 180.0f0)
        t = abs((2.0f0 * proj_vel * sin_value)) / K_G_VALUE
        range[i] = abs(proj_vel * t * cos_value)
        total_time[i] = t
        max_height[i] = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f0 * K_G_VALUE
    end
    return range, total_time, max_height
end

function main(args)
    if length(args) != 1
        println("Usage: ./main <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])

    rng = MersenneTwister(2)
    angle = Float32.(rand(rng, 10:99, NUM_ELEMENTS))
    velocity = Float32.(rand(rng, 10:409, NUM_ELEMENTS))
    d_angle = CuArray(angle)
    d_velocity = CuArray(velocity)
    d_range = CUDA.zeros(Float32, NUM_ELEMENTS)
    d_total_time = CUDA.zeros(Float32, NUM_ELEMENTS)
    d_max_height = CUDA.zeros(Float32, NUM_ELEMENTS)

    blocks = cld(NUM_ELEMENTS, BLOCK_SIZE)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=BLOCK_SIZE blocks=blocks calculate_range_kernel!(
            d_angle, d_velocity, d_range, d_total_time, d_max_height, Int32(NUM_ELEMENTS))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - start) * 1.0e-9 / repeat
    @printf("Average kernel execution time: %f (s)\n", elapsed_s)

    range = Array(d_range)
    total_time = Array(d_total_time)
    max_height = Array(d_max_height)
    ref_range, ref_total_time, ref_max_height = reference(angle, velocity)

    ok = all(abs.(range .- ref_range) .<= 1.0f0) &&
         all(abs.(total_time .- ref_total_time) .<= 1.0f0) &&
         all(abs.(max_height .- ref_max_height) .<= 1.0f0)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
