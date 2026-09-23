using CUDA
using Printf

const THREADS = 256

function drnd!(state::Base.RefValue{Int32})
    last = state[]
    state[] = Int32((Int64(1103515245) * Int64(state[]) + 12345) & 0x7fffffff)
    return Float64(last) / 2147483648.0
end

function initialize_bodies(nbodies::Int)
    rng = Ref(Int32(7))
    mass = fill(Float32(1 / nbodies), nbodies)
    px = Vector{Float32}(undef, nbodies)
    py = Vector{Float32}(undef, nbodies)
    pz = Vector{Float32}(undef, nbodies)
    vx = Vector{Float32}(undef, nbodies)
    vy = Vector{Float32}(undef, nbodies)
    vz = Vector{Float32}(undef, nbodies)
    ax = zeros(Float32, nbodies)
    ay = zeros(Float32, nbodies)
    az = zeros(Float32, nbodies)

    rsc = (3.0 * pi) / 16.0
    vsc = sqrt(1.0 / rsc)
    for i in 1:nbodies
        r = 1.0 / sqrt((drnd!(rng) * 0.999)^(-2.0 / 3.0) - 1.0)
        local x::Float64, y::Float64, z::Float64, sq::Float64
        while true
            x = drnd!(rng) * 2.0 - 1.0
            y = drnd!(rng) * 2.0 - 1.0
            z = drnd!(rng) * 2.0 - 1.0
            sq = x * x + y * y + z * z
            sq <= 1.0 && break
        end
        scale = rsc * r / sqrt(sq)
        px[i] = Float32(x * scale)
        py[i] = Float32(y * scale)
        pz[i] = Float32(z * scale)

        local v::Float64
        while true
            x = drnd!(rng)
            y = drnd!(rng) * 0.1
            y <= x * x * (1.0 - x * x)^3.5 && break
        end
        v = x * sqrt(2.0 / sqrt(1.0 + r * r))
        while true
            x = drnd!(rng) * 2.0 - 1.0
            y = drnd!(rng) * 2.0 - 1.0
            z = drnd!(rng) * 2.0 - 1.0
            sq = x * x + y * y + z * z
            sq <= 1.0 && break
        end
        scale = vsc * v / sqrt(sq)
        vx[i] = Float32(x * scale)
        vy[i] = Float32(y * scale)
        vz[i] = Float32(z * scale)
    end
    return px, py, pz, mass, vx, vy, vz, ax, ay, az
end

function direct_step_kernel!(px, py, pz, mass, vx, vy, vz, ax, ay, az,
                             nbodies::Int32, dtime::Float32, dthf::Float32,
                             epssq::Float32, step::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= nbodies
        pix = px[i]
        piy = py[i]
        piz = pz[i]
        fax = 0.0f0
        fay = 0.0f0
        faz = 0.0f0
        @inbounds for j in Int32(1):nbodies
            dx = px[j] - pix
            dy = py[j] - piy
            dz = pz[j] - piz
            dist2 = dx * dx + dy * dy + dz * dz + epssq
            inv = 1.0f0 / sqrt(dist2)
            s = mass[j] * inv * inv * inv
            fax += dx * s
            fay += dy * s
            faz += dz * s
        end

        vxi = vx[i]
        vyi = vy[i]
        vzi = vz[i]
        axi = ax[i]
        ayi = ay[i]
        azi = az[i]
        if step > Int32(0)
            vxi += (fax - axi) * dthf
            vyi += (fay - ayi) * dthf
            vzi += (faz - azi) * dthf
        end

        velhx = vxi + fax * dthf
        velhy = vyi + fay * dthf
        velhz = vzi + faz * dthf
        px[i] = pix + velhx * dtime
        py[i] = piy + velhy * dtime
        pz[i] = piz + velhz * dtime
        vx[i] = velhx + fax * dthf
        vy[i] = velhy + fay * dthf
        vz[i] = velhz + faz * dthf
        ax[i] = fax
        ay[i] = fay
        az[i] = faz
        i += stride
    end
    return
end

function main()
    println("ECL-BH v4.5")
    println("Copyright (c) 2010-2020 Texas State University")

    if length(ARGS) != 2
        println(stderr, "\narguments: number_of_bodies number_of_timesteps")
        return 1
    end

    nbodies = parse(Int, ARGS[1])
    timesteps = parse(Int, ARGS[2])
    if nbodies < 1 || nbodies > (1 << 30) || timesteps < 0
        return 1
    end

    @printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps)

    px, py, pz, mass, vx, vy, vz, ax, ay, az = initialize_bodies(nbodies)
    d_px = CuArray(px)
    d_py = CuArray(py)
    d_pz = CuArray(pz)
    d_mass = CuArray(mass)
    d_vx = CuArray(vx)
    d_vy = CuArray(vy)
    d_vz = CuArray(vz)
    d_ax = CuArray(ax)
    d_ay = CuArray(ay)
    d_az = CuArray(az)

    dtime = Float32(0.025)
    dthf = dtime * Float32(0.5)
    epssq = Float32(0.05 * 0.05)
    blocks = cld(nbodies, THREADS)

    CUDA.synchronize()
    start = time_ns()
    for step in 0:(timesteps - 1)
        @cuda threads=THREADS blocks=blocks direct_step_kernel!(
            d_px, d_py, d_pz, d_mass, d_vx, d_vy, d_vz, d_ax, d_ay, d_az,
            Int32(nbodies), dtime, dthf, epssq, Int32(step))
    end
    CUDA.synchronize()
    runtime = (time_ns() - start) * 1e-9
    @printf("Total kernel execution time: %.4f s\n", runtime)
    return 0
end

exit(main())
