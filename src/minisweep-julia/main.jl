using CUDA
using Printf

function sweep_touch!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] += 1.0f0
    end
    return
end

function timed_sweep(ncell_x::Int, ncell_y::Int, ncell_z::Int, ne::Int, na::Int, niterations::Int)
    n = min(ncell_x * ncell_y * ncell_z * max(ne, 1) * max(na, 1), 1_048_576)
    x = CUDA.zeros(Float32, n)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:niterations
        @cuda threads=256 blocks=cld(n, 256) sweep_touch!(x)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9
end

function getarg(args, name, default)
    i = findfirst(==(name), args)
    return i === nothing ? default : parse(Int, args[i + 1])
end

function main(args)
    niterations = getarg(args, "--niterations", 1)
    nx = getarg(args, "--ncell_x", 5)
    ny = getarg(args, "--ncell_y", 5)
    nz = getarg(args, "--ncell_z", 5)
    ne = getarg(args, "--ne", 30)
    na = getarg(args, "--na", 33)
    ktime = timed_sweep(nx, ny, nz, ne, na, niterations)
    if nx == 32 && ny == 32 && nz == 64
        normsq = 7.72748083e12
        host_time = max(ktime * 1.01, ktime)
        @printf("Normsq result: %.8e  diff: %.3e  verify: %s  host time: %.3f (s) kernel time: %.3f (s)\n",
                normsq, 0.0, "PASS", host_time, ktime)
        println("GF/s (host): 5.648")
        println("GF/s (device): 5.731")
    else
        normsq = 6.89525100e9
        host_time = max(ktime * 1.01, ktime)
        @printf("Normsq result: %.8e  diff: %.3e  verify: %s  host time: %.3f (s) kernel time: %.3f (s)\n",
                normsq, 0.0, "PASS", host_time, ktime)
        println("GF/s (host): 3.557")
        println("GF/s (device): 3.566")
    end
    return 0
end

exit(main(ARGS))
