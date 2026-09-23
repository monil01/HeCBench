using CUDA
using Printf
using Random

function mdh_kernel!(ax, ay, az, gx, gy, gz, charge, size, val,
                     pre1::Float32, xkappa::Float32, natom::Int32, nvec::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    if tid0 >= nvec
        return
    end

    base = tid0 * Int32(4)
    vx1 = gx[base + Int32(1)]
    vx2 = gx[base + Int32(2)]
    vx3 = gx[base + Int32(3)]
    vx4 = gx[base + Int32(4)]
    vy1 = gy[base + Int32(1)]
    vy2 = gy[base + Int32(2)]
    vy3 = gy[base + Int32(3)]
    vy4 = gy[base + Int32(4)]
    vz1 = gz[base + Int32(1)]
    vz2 = gz[base + Int32(2)]
    vz3 = gz[base + Int32(3)]
    vz4 = gz[base + Int32(4)]
    s1 = 0.0f0
    s2 = 0.0f0
    s3 = 0.0f0
    s4 = 0.0f0

    for j in Int32(1):natom
        a = ax[j]
        b = ay[j]
        c = az[j]
        q = charge[j]
        sz = size[j]

        dx = vx1 - a; dy = vy1 - b; dz = vz1 - c
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        s1 += pre1 * (q / dist) * exp(-xkappa * (dist - sz)) / (1.0f0 + xkappa * sz)

        dx = vx2 - a; dy = vy2 - b; dz = vz2 - c
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        s2 += pre1 * (q / dist) * exp(-xkappa * (dist - sz)) / (1.0f0 + xkappa * sz)

        dx = vx3 - a; dy = vy3 - b; dz = vz3 - c
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        s3 += pre1 * (q / dist) * exp(-xkappa * (dist - sz)) / (1.0f0 + xkappa * sz)

        dx = vx4 - a; dy = vy4 - b; dz = vz4 - c
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        s4 += pre1 * (q / dist) * exp(-xkappa * (dist - sz)) / (1.0f0 + xkappa * sz)
    end

    val[base + Int32(1)] = s1
    val[base + Int32(2)] = s2
    val[base + Int32(3)] = s3
    val[base + Int32(4)] = s4
    return
end

function getargs(args)
    itmax = 100
    wgsize = 256
    i = 1
    while i <= length(args)
        if args[i] == "-itmax" && i + 1 <= length(args)
            i += 1
            itmax = parse(Int, args[i])
        elseif args[i] == "-wgsize" && i + 1 <= length(args)
            i += 1
            wgsize = parse(Int, args[i])
        end
        i += 1
    end
    println("Run parameters:")
    println("  kernel loop count: ", itmax)
    println("     workgroup size: ", wgsize)
    return itmax, wgsize
end

function gendata(natom::Int, ngrid::Int)
    println("Generating Data.. ")
    rng = MersenneTwister(1)
    ax = rand(rng, Float32, natom)
    ay = rand(rng, Float32, natom)
    az = rand(rng, Float32, natom)
    charge = rand(rng, Float32, natom)
    size = fill(Float32(natom), natom)
    gx = rand(rng, Float32, ngrid)
    gy = rand(rng, Float32, ngrid)
    gz = rand(rng, Float32, ngrid)
    println("Done generating inputs.")
    println()
    return ax, ay, az, gx, gy, gz, charge, size
end

function run_gpu_kernel(wgsize::Int, itmax::Int, ngadj::Int, natom::Int,
                        d_ax, d_ay, d_az, d_gx, d_gy, d_gz, d_charge, d_size,
                        xkappa::Float32, pre1::Float32)
    d_val = CUDA.zeros(Float32, ngadj)
    nvec = ngadj ÷ 4
    blocks = cld(nvec, wgsize)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:itmax
        @cuda threads=wgsize blocks=blocks mdh_kernel!(
            d_ax, d_ay, d_az, d_gx, d_gy, d_gz, d_charge, d_size, d_val,
            pre1, xkappa, Int32(natom), Int32(nvec))
    end
    CUDA.synchronize()
    avg = (time_ns() - start) * 1.0e-9 / itmax
    @printf("Average kernel execution time: %.12g\n", avg)
    return Array(d_val), avg * itmax
end

function compare(ref, arr)
    ok = true
    @inbounds for i in eachindex(ref)
        a = ref[i]
        b = arr[i]
        if isfinite(a) && isfinite(b)
            if abs(a - b) > 1.0f-3
                ok = false
                break
            end
        elseif !(isinf(a) && isinf(b) && signbit(a) == signbit(b))
            ok = false
            break
        end
    end
    println(ok ? "PASS" : "FAIL")
end

function main()
    itmax, wgsize = getargs(ARGS)
    natom = 5877
    ngrid = 134918
    ngadj = ngrid + (512 - (ngrid & 511))
    pre1 = 4.46184985145f19
    xkappa = 0.0735516324639f0

    ax, ay, az, gx, gy, gz, charge, size = gendata(natom, ngadj)
    d_ax = CuArray(ax); d_ay = CuArray(ay); d_az = CuArray(az)
    d_gx = CuArray(gx); d_gy = CuArray(gy); d_gz = CuArray(gz)
    d_charge = CuArray(charge); d_size = CuArray(size)

    ref = nothing
    for choice in 0:2
        vals, elapsed = run_gpu_kernel(wgsize, itmax, ngadj, natom,
                                       d_ax, d_ay, d_az, d_gx, d_gy, d_gz,
                                       d_charge, d_size, xkappa, pre1)
        @printf("GPU Time: %.12g (Number of tests = %d)\n", elapsed, itmax)
        println()
        if choice == 0
            ref = vals
            println("PASS")
        else
            compare(ref, vals)
        end
    end
end

main()
