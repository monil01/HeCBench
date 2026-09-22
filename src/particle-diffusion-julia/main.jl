using CUDA
using Printf
using Random

const GRID_SIZE = 21
const N_PARTICLES = 147456
const RADIUS = 0.5f0

function simulation_kernel!(particle_x, particle_y, random_x, random_y, visits,
                            n_particles::Int32, niterations::Int32,
                            grid_size::Int32, radius::Float32)
    ii0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if ii0 >= n_particles
        return
    end

    px = @inbounds particle_x[ii0 + Int32(1)]
    py = @inbounds particle_y[ii0 + Int32(1)]
    map_base = ii0 * grid_size * grid_size
    iter = Int32(0)
    while iter < niterations
        randx = @inbounds random_x[iter * n_particles + ii0 + Int32(1)]
        randy = @inbounds random_y[iter * n_particles + ii0 + Int32(1)]
        px += randx / 1000.0f0 - 0.0495f0
        py += randy / 1000.0f0 - 0.0495f0

        dx = px - trunc(px)
        dy = py - trunc(py)
        ix = Int32(floor(px))
        iy = Int32(floor(py))

        if px < Float32(grid_size) && py < Float32(grid_size) && px >= 0.0f0 && py >= 0.0f0
            if dx * dx + dy * dy <= radius * radius
                @inbounds visits[map_base + iy * grid_size + ix + Int32(1)] += UInt64(1)
            end
        end
        iter += Int32(1)
    end

    @inbounds particle_x[ii0 + Int32(1)] = px
    @inbounds particle_y[ii0 + Int32(1)] = py
    return
end

function reference!(particle_x, particle_y, random_x, random_y, map,
                    n_particles::Int, niterations::Int, grid_size::Int,
                    radius::Float32)
    for ii in 0:(n_particles - 1)
        px = particle_x[ii + 1]
        py = particle_y[ii + 1]
        map_base = ii * grid_size * grid_size
        for iter in 0:(niterations - 1)
            randx = random_x[iter * n_particles + ii + 1]
            randy = random_y[iter * n_particles + ii + 1]
            px += randx / 1000.0f0 - 0.0495f0
            py += randy / 1000.0f0 - 0.0495f0
            dx = px - trunc(px)
            dy = py - trunc(py)
            ix = floor(Int, px)
            iy = floor(Int, py)
            if px < grid_size && py < grid_size && px >= 0.0f0 && py >= 0.0f0
                if dx * dx + dy * dy <= radius * radius
                    map[map_base + iy * grid_size + ix + 1] += UInt64(1)
                end
            end
        end
        particle_x[ii + 1] = px
        particle_y[ii + 1] = py
    end
end

function main(args)
    if length(args) != 2
        println(" Incorrect number of parameters ")
        println(" Usage: main.jl <Number of iterations within the kernel> <Kernel execution count>")
        return 1
    end

    niterations = parse(Int, args[1])
    nrepeat = parse(Int, args[2])
    n_particles = N_PARTICLES
    grid_size = GRID_SIZE
    map_size = n_particles * grid_size * grid_size

    particle_x = fill(10.0f0, n_particles)
    particle_y = fill(10.0f0, n_particles)
    rng = MersenneTwister(17)
    random_x = Float32.(rand(rng, 0:99, n_particles * niterations))
    random_y = Float32.(rand(rng, 0:99, n_particles * niterations))
    visits = zeros(UInt64, map_size)
    map_ref = zeros(UInt64, map_size)

    dev = CUDA.device()
    println(" Running on $(CUDA.name(dev))")
    println(" The device max work-group size is $(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))")
    println(" The number of iterations is $niterations")
    println(" The number of kernel execution is $nrepeat")
    println(" The number of particles is $n_particles")

    d_random_x = CuArray(random_x)
    d_random_y = CuArray(random_y)
    d_particle_x = CuArray(particle_x)
    d_particle_y = CuArray(particle_y)
    d_visits = CuArray(visits)

    threads = 256
    blocks = cld(n_particles, threads)
    time_total = 0.0
    for _ in 1:nrepeat
        copyto!(d_particle_x, particle_x)
        copyto!(d_particle_y, particle_y)
        CUDA.fill!(d_visits, UInt64(0))
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks simulation_kernel!(
            d_particle_x, d_particle_y, d_random_x, d_random_y, d_visits,
            Int32(n_particles), Int32(niterations), Int32(grid_size), RADIUS)
        CUDA.synchronize()
        time_total += time_ns() - start
    end

    println()
    @printf("Average kernel execution time: %f (s)\n", (time_total * 1.0e-9) / nrepeat)
    copyto!(visits, Array(d_visits))
    println()
    @printf("Simulation time: %f (s) \n", time_total * 1.0e-9)

    ref_px = copy(particle_x)
    ref_py = copy(particle_y)
    reference!(ref_px, ref_py, random_x, random_y, map_ref, n_particles,
               niterations, grid_size, RADIUS)
    mismatches = count(i -> visits[i] != map_ref[i], eachindex(visits))
    println(mismatches <= 2 ? "PASS" : "FAIL")
    return mismatches <= 2 ? 0 : 1
end

exit(main(ARGS))
