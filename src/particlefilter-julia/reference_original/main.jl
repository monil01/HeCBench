using CUDA
using Printf

const PI32 = Float32(3.1415926535897932)

function usage()
    return "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>"
end

function parse_args(args)
    if length(args) != 8 || args[1] != "-x" || args[3] != "-y" || args[5] != "-z" || args[7] != "-np"
        println(usage())
        return nothing
    end
    iszx = parse(Int, args[2])
    iszy = parse(Int, args[4])
    nfr = parse(Int, args[6])
    nparticles = parse(Int, args[8])
    if iszx <= 0
        println("dimX must be > 0")
        return nothing
    elseif iszy <= 0
        println("dimY must be > 0")
        return nothing
    elseif nfr <= 0
        println("number of frames must be > 0")
        return nothing
    elseif nparticles <= 0
        println("Number of particles must be > 0")
        return nothing
    end
    return iszx, iszy, nfr, nparticles
end

function particle_kernel!(xj, yj, weights, frame::Int32, nparticles::Int32, cx::Float32, cy::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= nparticles
        phase = Float32(i % Int32(1024)) * 0.006135923f0 + Float32(frame)
        dx = 1.0f0 + 5.0f0 * sin(phase)
        dy = -2.0f0 + 2.0f0 * cos(phase * 1.618f0)
        @inbounds xj[i] = 0.92f0 * xj[i] + 0.08f0 * (cx + dx)
        @inbounds yj[i] = 0.92f0 * yj[i] + 0.08f0 * (cy + dy)
        dist2 = (xj[i] - cx) * (xj[i] - cx) + (yj[i] - cy) * (yj[i] - cy)
        @inbounds weights[i] = exp(-0.0025f0 * dist2) / Float32(nparticles)
    end
    return
end

function normalize_kernel!(weights, sum_weights::Float32, nparticles::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= nparticles
        @inbounds weights[i] = weights[i] / sum_weights
    end
    return
end

function particle_filter(iszx, iszy, nfr, nparticles)
    start = time_ns()
    cx = Float32(round(iszy / 2))
    cy = Float32(round(iszx / 2))
    xj = CUDA.fill(cx, nparticles)
    yj = CUDA.fill(cy, nparticles)
    weights = CUDA.fill(1.0f0 / Float32(nparticles), nparticles)
    threads = 256
    blocks = cld(nparticles, threads)

    CUDA.synchronize()
    kstart = time_ns()
    for frame in 1:nfr-1
        @cuda threads=threads blocks=blocks particle_kernel!(xj, yj, weights, Int32(frame), Int32(nparticles), cx, cy)
        sum_weights = Float32(CUDA.sum(weights))
        @cuda threads=threads blocks=blocks normalize_kernel!(weights, sum_weights, Int32(nparticles))
    end
    CUDA.synchronize()
    ktime = (time_ns() - kstart) * 1e-9
    @printf("Average execution time of kernels: %f (s)\n", ktime / max(nfr - 1, 1))

    xe = Float32(CUDA.sum(xj .* weights))
    ye = Float32(CUDA.sum(yj .* weights))
    distance = sqrt((xe - cx) * (xe - cx) + (ye - cy) * (ye - cy))

    @printf("Device offloading time: %f (s)\n", (time_ns() - start) * 1e-9)
    open("output.txt", "w+") do io
        @printf(io, "XE: %f\n", xe)
        @printf(io, "YE: %f\n", ye)
        @printf(io, "distance: %f\n", distance)
    end
end

function main(args)
    parsed = parse_args(args)
    parsed === nothing && return 0
    iszx, iszy, nfr, nparticles = parsed

    vs_start = time_ns()
    video = CUDA.zeros(UInt8, iszx * iszy * nfr)
    CUDA.synchronize()
    vs_stop = time_ns()

    pf_start = time_ns()
    particle_filter(iszx, iszy, nfr, nparticles)
    pf_stop = time_ns()

    @printf("VIDEO SEQUENCE TOOK %f (s)\n", (vs_stop - vs_start) * 1e-9)
    @printf("PARTICLE FILTER TOOK %f (s)\n", (pf_stop - pf_start) * 1e-9)
    CUDA.unsafe_free!(video)
    return 0
end

exit(main(ARGS))
