using CUDA
using Printf
using Random

const CHUNK_S = 4096

function cmpfhd!(rmu, imu, rfhd, ifhd, x, y, z, kx, ky, kz, samples::Int32, voxels::Int32)
    n0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if n0 < samples
        idx = n0 + Int32(1)
        @inbounds begin
            xn = x[idx]
            yn = y[idx]
            zn = z[idx]
            rfhdn = rfhd[idx]
            ifhdn = ifhd[idx]
        end
        for m0 in Int32(0):(voxels - Int32(1))
            midx = m0 + Int32(1)
            @inbounds e = Float32(2π) * (kx[midx] * xn + ky[midx] * yn + kz[midx] * zn)
            c = cos(e)
            s = sin(e)
            @inbounds begin
                rm = rmu[midx]
                im = imu[midx]
            end
            rfhdn += rm * c - im * s
            ifhdn += im * c + rm * s
        end
        @inbounds begin
            rfhd[idx] = rfhdn
            ifhd[idx] = ifhdn
        end
    end
    return
end

function init_inputs(samples::Int, voxels::Int)
    rng = MersenneTwister(2)
    h_rfhd = Vector{Float32}(undef, samples)
    h_ifhd = Vector{Float32}(undef, samples)
    h_x = Vector{Float32}(undef, samples)
    h_y = Vector{Float32}(undef, samples)
    h_z = Vector{Float32}(undef, samples)
    for i in 1:samples
        h_rfhd[i] = Float32(i - 1) / Float32(samples)
        h_ifhd[i] = Float32(i - 1) / Float32(samples)
        h_x[i] = Float32(0.3) + (rand(rng, Bool) ? Float32(0.1) : Float32(-0.1))
        h_y[i] = Float32(0.2) + (rand(rng, Bool) ? Float32(0.1) : Float32(-0.1))
        h_z[i] = Float32(0.1) + (rand(rng, Bool) ? Float32(0.1) : Float32(-0.1))
    end

    h_rmu = Vector{Float32}(undef, voxels)
    h_imu = Vector{Float32}(undef, voxels)
    h_kx = Vector{Float32}(undef, voxels)
    h_ky = Vector{Float32}(undef, voxels)
    h_kz = Vector{Float32}(undef, voxels)
    for i in 1:voxels
        h_rmu[i] = Float32(i - 1) / Float32(voxels)
        h_imu[i] = Float32(i - 1) / Float32(voxels)
        h_kx[i] = Float32(0.1) + (rand(rng, Bool) ? Float32(0.1) : Float32(-0.1))
        h_ky[i] = Float32(0.2) + (rand(rng, Bool) ? Float32(0.1) : Float32(-0.1))
        h_kz[i] = Float32(0.3) + (rand(rng, Bool) ? Float32(0.1) : Float32(-0.1))
    end
    return h_rmu, h_imu, h_kx, h_ky, h_kz, h_rfhd, h_ifhd, h_x, h_y, h_z
end

function host_reference!(h_rmu, h_imu, h_kx, h_ky, h_kz, h_rfhd, h_ifhd, h_x, h_y, h_z)
    samples = length(h_x)
    voxels = length(h_rmu)
    for n in 1:samples
        r = h_rfhd[n]
        imacc = h_ifhd[n]
        xn = h_x[n]
        yn = h_y[n]
        zn = h_z[n]
        for m in 1:voxels
            e = Float32(2π) * (h_kx[m] * xn + h_ky[m] * yn + h_kz[m] * zn)
            c = cos(e)
            s = sin(e)
            r += h_rmu[m] * c - h_imu[m] * s
            imacc += h_imu[m] * c + h_rmu[m] * s
        end
        h_rfhd[n] = r
        h_ifhd[n] = imacc
    end
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <#samples> <#voxels> <verify>")
        return 1
    end

    samples = parse(Int, args[1])
    voxels = parse(Int, args[2])
    verify = parse(Int, args[3])

    h_rmu, h_imu, h_kx, h_ky, h_kz, h_rfhd, h_ifhd, h_x, h_y, h_z =
        init_inputs(samples, voxels)
    rfhd0 = copy(h_rfhd)
    ifhd0 = copy(h_ifhd)

    println("Run FHd on a device")
    d_rfhd = CuArray(h_rfhd)
    d_ifhd = CuArray(h_ifhd)
    d_x = CuArray(h_x)
    d_y = CuArray(h_y)
    d_z = CuArray(h_z)

    ntpb = 256
    nblks = cld(samples, ntpb)
    nchunks = cld(voxels, CHUNK_S)

    CUDA.synchronize()
    t0 = time_ns()
    for chunk0 in 0:(nchunks - 1)
        first = chunk0 * CHUNK_S + 1
        last = min(voxels, first + CHUNK_S - 1)
        d_rmu = CuArray(@view h_rmu[first:last])
        d_imu = CuArray(@view h_imu[first:last])
        d_kx = CuArray(@view h_kx[first:last])
        d_ky = CuArray(@view h_ky[first:last])
        d_kz = CuArray(@view h_kz[first:last])
        @cuda threads=ntpb blocks=nblks cmpfhd!(d_rmu, d_imu, d_rfhd, d_ifhd,
                                                d_x, d_y, d_z, d_kx, d_ky, d_kz,
                                                Int32(samples), Int32(last - first + 1))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1.0e-9
    @printf("Device execution time %f (s)\n", elapsed_s)

    rfhd = Array(d_rfhd)
    ifhd = Array(d_ifhd)

    if verify != 0
        println("Computing root mean square error between host and device results.")
        println("This will take a while..")
        host_reference!(h_rmu, h_imu, h_kx, h_ky, h_kz, rfhd0, ifhd0, h_x, h_y, h_z)
        err = Float32(0)
        for i in 1:samples
            err += (rfhd0[i] - rfhd[i]) * (rfhd0[i] - rfhd[i]) +
                   (ifhd0[i] - ifhd[i]) * (ifhd0[i] - ifhd[i])
        end
        rmse = sqrt(err / Float32(2 * samples))
        @printf("RMSE = %f\n", rmse)
        println(rmse <= Float32(1.0e-3) ? "PASS" : "FAIL")
    end

    return 0
end

exit(main(ARGS))
