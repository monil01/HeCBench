using CUDA
using Printf

const GRID_SIZE = 64
const NUM_PARTICLES = 16384
const TIMESTEP = Float32(0.5)

function integrate_kernel!(pos, vel, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        base = (i - Int32(1)) * Int32(4)
        @inbounds begin
            vel[base + 2] += Float32(-0.0003) * TIMESTEP
            pos[base + 1] += vel[base + 1] * TIMESTEP
            pos[base + 2] += vel[base + 2] * TIMESTEP
            pos[base + 3] += vel[base + 3] * TIMESTEP
        end
        i += stride
    end
    return
end

function init_grid(num_particles::Int)
    pos = Vector{Float32}(undef, 4 * num_particles)
    vel = zeros(Float32, 4 * num_particles)
    s = ceil(Int, num_particles^(1 / 3))
    radius = Float32(0.023)
    spacing = radius * Float32(2)
    idx = 1
    for z in 0:s-1, y in 0:s-1, x in 0:s-1
        idx > num_particles && break
        base = 4 * (idx - 1)
        pos[base + 1] = spacing * x + radius - Float32(1)
        pos[base + 2] = spacing * y + radius - Float32(1)
        pos[base + 3] = spacing * z + radius - Float32(1)
        pos[base + 4] = Float32(1)
        idx += 1
    end
    return pos, vel
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <iterations>")
        return 1
    end
    iterations = parse(Int, ARGS[1])
    num_cells = GRID_SIZE * GRID_SIZE * GRID_SIZE

    @printf(" grid: %d x %d x %d = %d cells\n", GRID_SIZE, GRID_SIZE, GRID_SIZE, num_cells)
    @printf(" particles: %d\n\n", NUM_PARTICLES)

    h_pos, h_vel = init_grid(NUM_PARTICLES)
    d_pos = CuArray(h_pos)
    d_vel = CuArray(h_vel)
    threads = 256
    blocks = cld(NUM_PARTICLES, threads)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks integrate_kernel!(d_pos, d_vel, Int32(NUM_PARTICLES))
    end
    CUDA.synchronize()
    seconds = (time_ns() - t0) * 1e-9
    @printf("Total execution time of %d loop iterations: %f (s)\n", iterations, seconds)
    @printf("Average execution time of a loop iteration: %f (us)\n", seconds * 1e6 / iterations)
    return 0
end

exit(main())
