using CUDA
using Printf
using Random

const RealType = Float32

function accelerate_particles!(posx, posy, posz, accx, accy, accz, mass,
                               n::Int32, softening::Float32, grav::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end

    idx = i
    pix = posx[idx]
    piy = posy[idx]
    piz = posz[idx]
    ax = accx[idx]
    ay = accy[idx]
    az = accz[idx]

    for j in Int32(1):n
        dx = posx[j] - pix
        dy = posy[j] - piy
        dz = posz[j] - piz
        distance_sqr = dx * dx + dy * dy + dz * dz + softening
        distance_inv = inv(sqrt(distance_sqr))
        strength = grav * mass[j] * distance_inv * distance_inv * distance_inv
        ax += dx * strength
        ay += dy * strength
        az += dz * strength
    end

    accx[idx] = ax
    accy[idx] = ay
    accz[idx] = az
    return
end

function update_particles!(posx, posy, posz, velx, vely, velz, accx, accy, accz,
                           mass, energy, n::Int32, dt::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > n
        return
    end

    vx = velx[i] + accx[i] * dt
    vy = vely[i] + accy[i] * dt
    vz = velz[i] + accz[i] * dt

    velx[i] = vx
    vely[i] = vy
    velz[i] = vz
    posx[i] += vx * dt
    posy[i] += vy * dt
    posz[i] += vz * dt
    accx[i] = 0.0f0
    accy[i] = 0.0f0
    accz[i] = 0.0f0
    energy[i] = mass[i] * (vx * vx + vy * vy + vz * vz)
    return
end

function init_particles(n::Int)
    rng = MersenneTwister(42)
    posx = rand(rng, RealType, n)
    posy = rand(rng, RealType, n)
    posz = rand(rng, RealType, n)

    rng = MersenneTwister(42)
    velx = (rand(rng, RealType, n) .* 2.0f0 .- 1.0f0) .* 1.0f-3
    vely = (rand(rng, RealType, n) .* 2.0f0 .- 1.0f0) .* 1.0f-3
    velz = (rand(rng, RealType, n) .* 2.0f0 .- 1.0f0) .* 1.0f-3

    accx = zeros(RealType, n)
    accy = zeros(RealType, n)
    accz = zeros(RealType, n)

    rng = MersenneTwister(42)
    mass = RealType(n) .* rand(rng, RealType, n)
    return posx, posy, posz, velx, vely, velz, accx, accy, accz, mass
end

function step_cpu!(posx, posy, posz, velx, vely, velz, accx, accy, accz, mass,
                   energy, dt::Float32)
    n = length(posx)
    softening = 1.0f-3
    grav = 6.67259f-11

    for i in 1:n
        pix = posx[i]
        piy = posy[i]
        piz = posz[i]
        ax = accx[i]
        ay = accy[i]
        az = accz[i]
        for j in 1:n
            dx = posx[j] - pix
            dy = posy[j] - piy
            dz = posz[j] - piz
            distance_sqr = dx * dx + dy * dy + dz * dz + softening
            distance_inv = inv(sqrt(distance_sqr))
            strength = grav * mass[j] * distance_inv * distance_inv * distance_inv
            ax += dx * strength
            ay += dy * strength
            az += dz * strength
        end
        accx[i] = ax
        accy[i] = ay
        accz[i] = az
    end

    for i in 1:n
        velx[i] += accx[i] * dt
        vely[i] += accy[i] * dt
        velz[i] += accz[i] * dt
        posx[i] += velx[i] * dt
        posy[i] += vely[i] * dt
        posz[i] += velz[i] * dt
        accx[i] = 0.0f0
        accy[i] = 0.0f0
        accz[i] = 0.0f0
        energy[i] = mass[i] * (velx[i] * velx[i] + vely[i] * vely[i] + velz[i] * velz[i])
    end
    return 0.5f0 * sum(energy)
end

function run_gpu(n::Int, nsteps::Int)
    posx, posy, posz, velx, vely, velz, accx, accy, accz, mass = init_particles(n)
    energy = zeros(RealType, n)

    d_posx = CuArray(posx); d_posy = CuArray(posy); d_posz = CuArray(posz)
    d_velx = CuArray(velx); d_vely = CuArray(vely); d_velz = CuArray(velz)
    d_accx = CuArray(accx); d_accy = CuArray(accy); d_accz = CuArray(accz)
    d_mass = CuArray(mass); d_energy = CuArray(energy)

    threads = 256
    blocks = cld(n, threads)
    dt = 0.1f0
    gflops = 1.0e-9 * ((11.0 + 10.0) * n * n + n * 19.0)
    nf = 0
    av = 0.0
    dev = 0.0
    kenergy = 0.0f0

    CUDA.synchronize()
    total_start = time_ns()
    for s in 1:nsteps
        step_start = time_ns()
        @cuda threads=threads blocks=blocks accelerate_particles!(
            d_posx, d_posy, d_posz, d_accx, d_accy, d_accz, d_mass,
            Int32(n), 1.0f-3, 6.67259f-11)
        @cuda threads=threads blocks=blocks update_particles!(
            d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, d_accx, d_accy, d_accz,
            d_mass, d_energy, Int32(n), dt)
        CUDA.synchronize()
        elapsed_seconds = (time_ns() - step_start) * 1.0e-9
        energy_host = Array(d_energy)
        kenergy = 0.5f0 * sum(energy_host)
        nf += 1
        if nf > 2
            perf = gflops / elapsed_seconds
            av += perf
            dev += perf * perf
        end
    end
    total_time = (time_ns() - total_start) * 1.0e-9
    av /= (nf - 2)
    dev = (nf == 3) ? 0.0 : sqrt(dev / (nf - 2) - av * av)

    println()
    println("# Total Energy        : ", kenergy)
    println("# Total Time (s)      : ", total_time)
    println("# Average Performance : ", av, " +- ", dev)
    println("===============================")
    return kenergy
end

function run_cpu_reference(n::Int, nsteps::Int)
    posx, posy, posz, velx, vely, velz, accx, accy, accz, mass = init_particles(n)
    energy = zeros(RealType, n)
    kenergy = 0.0f0
    for _ in 1:nsteps
        kenergy = step_cpu!(posx, posy, posz, velx, vely, velz, accx, accy, accz,
                            mass, energy, 0.1f0)
    end
    return kenergy
end

function main()
    n = 16000
    nsteps = 10
    if length(ARGS) > 0
        n = parse(Int, ARGS[1])
        if length(ARGS) == 2
            nsteps = parse(Int, ARGS[2])
            if nsteps < 3
                println(stderr, "The number of integration steps should be at least 3.")
                exit(1)
            end
        end
    end

    println("===============================")
    println(" Initialize Gravity Simulation")
    kenergy = run_gpu(n, nsteps)
    ref = run_cpu_reference(n, nsteps)
    println()
    println(abs(kenergy - ref) < 1.0f-3 ? "PASS" : "FAIL")
end

main()
