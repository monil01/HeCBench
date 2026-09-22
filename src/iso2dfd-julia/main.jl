using CUDA
using Printf

const BLOCK_SIZE = 16
const DT = 0.002f0
const DXY = 20.0f0
const HALF_LENGTH = 1

function initialize!(prev::Vector{Float32}, next::Vector{Float32}, vel::Vector{Float32},
                     nrows::Int, ncols::Int)
    println("Initializing ... ")
    wavelet = Float32[
        0.016387336, -0.041464937, -0.067372555, 0.386110067,
        0.812723635, 0.416998396, 0.076488599, -0.059434419,
        0.023680172, 0.005611435, 0.001823209, -0.000720549
    ]
    fill!(prev, 0.0f0)
    fill!(next, 0.0f0)
    fill!(vel, 2250000.0f0)

    for s in 11:-1:0
        for i in (nrows ÷ 2 - s):(nrows ÷ 2 + s - 1)
            offset = i * ncols
            for k in (ncols ÷ 2 - s):(ncols ÷ 2 + s - 1)
                prev[offset + k + 1] = wavelet[s + 1]
            end
        end
    end
end

function iso_2dfd_iteration_cpu!(next::Vector{Float32}, prev::Vector{Float32},
                                 vel::Vector{Float32}, dtdivdxy::Float32,
                                 nrows::Int, ncols::Int, niters::Int)
    for _ in 1:niters
        for i in 1:(nrows - HALF_LENGTH - 1)
            for j in 1:(ncols - HALF_LENGTH - 1)
                gid = j + i * ncols + 1
                value = 0.0f0
                value += prev[gid + 1] - 2.0f0 * prev[gid] + prev[gid - 1]
                value += prev[gid + ncols] - 2.0f0 * prev[gid] + prev[gid - ncols]
                value *= dtdivdxy * vel[gid]
                next[gid] = 2.0f0 * prev[gid] - next[gid] + value
            end
        end
        next, prev = prev, next
    end
    return next, prev
end

function iso_2dfd_kernel!(next, prev, vel, dtdivdxy::Float32, nrows::Int32, ncols::Int32)
    gid_col = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    gid_row = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    if gid_row < nrows && gid_col < ncols
        if gid_col >= Int32(HALF_LENGTH) && gid_col < ncols - Int32(HALF_LENGTH) &&
           gid_row >= Int32(HALF_LENGTH) && gid_row < nrows - Int32(HALF_LENGTH)
            gid = gid_col + gid_row * ncols + Int32(1)
            value = 0.0f0
            @inbounds begin
                value += prev[gid + Int32(1)] - 2.0f0 * prev[gid] + prev[gid - Int32(1)]
                value += prev[gid + ncols] - 2.0f0 * prev[gid] + prev[gid - ncols]
                value *= dtdivdxy * vel[gid]
                next[gid] = 2.0f0 * prev[gid] - next[gid] + value
            end
        end
    end
    return
end

function within_epsilon(output::Vector{Float32}, reference::Vector{Float32},
                        dimx::Int, dimy::Int, radius::Int, delta::Float32=0.1f0)
    error = false
    norm2 = 0.0
    open("error_diff.txt", "w") do fp
        for iy in 0:(dimy - 1), ix in 0:(dimx - 1)
            idx = iy * dimx + ix + 1
            if ix >= radius && ix < dimx - radius && iy >= radius && iy < dimy - radius
                difference = abs(reference[idx] - output[idx])
                norm2 += Float64(difference * difference)
                if difference > delta
                    error = true
                    @printf(fp, " ERROR: (%d,%d)\t%e instead of %e (|e|=%e)\n",
                            ix, iy, output[idx], reference[idx], difference)
                end
            end
        end
    end
    if error
        @printf("error (Euclidean norm): %.9e\n", sqrt(norm2))
    end
    return error
end

function main(args)
    if length(args) != 3
        println(" Incorrect parameters ")
        println(" Usage: main.jl n1 n2 Iterations ")
        return 1
    end

    nrows = parse(Int, args[1])
    ncols = parse(Int, args[2])
    niters = parse(Int, args[3])
    nsize = nrows * ncols

    prev_base = Vector{Float32}(undef, nsize)
    next_base = Vector{Float32}(undef, nsize)
    next_cpu = Vector{Float32}(undef, nsize)
    vel_base = Vector{Float32}(undef, nsize)
    dtdivdxy = (DT * DT) / (DXY * DXY)

    initialize!(prev_base, next_base, vel_base, nrows, ncols)
    println("Grid Sizes: $nrows $ncols")
    println("Iterations: $niters")
    println()
    println("Computing wavefield in device ..")
    dev = CUDA.device()
    println("Running on:: $(CUDA.name(dev))")
    println("The Device Max Work Group Size is : $(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))")

    d_next = CuArray(next_base)
    d_prev = CuArray(prev_base)
    d_vel = CuArray(vel_base)
    blocks = (cld(ncols, BLOCK_SIZE), cld(nrows, BLOCK_SIZE))
    threads = (BLOCK_SIZE, BLOCK_SIZE)

    CUDA.synchronize()
    kstart = time_ns()
    for k in 0:(niters - 1)
        if isodd(k)
            @cuda threads=threads blocks=blocks iso_2dfd_kernel!(d_prev, d_next, d_vel, dtdivdxy, Int32(nrows), Int32(ncols))
        else
            @cuda threads=threads blocks=blocks iso_2dfd_kernel!(d_next, d_prev, d_vel, dtdivdxy, Int32(nrows), Int32(ncols))
        end
    end
    CUDA.synchronize()
    elapsed_ns = time_ns() - kstart
    @printf("Total kernel execution time %f (ms)\n", elapsed_ns * 1.0e-6)
    @printf("Average kernel execution time %f (us)\n", (elapsed_ns * 1.0e-3) / niters)

    copyto!(next_base, Array(d_next))
    open("wavefield_snapshot.bin", "w") do io
        write(io, next_base)
    end

    println("Computing wavefield in CPU ..")
    initialize!(prev_base, next_cpu, vel_base, nrows, ncols)
    cpu_start = time_ns()
    iso_2dfd_iteration_cpu!(next_cpu, prev_base, vel_base, dtdivdxy, nrows, ncols, niters)
    cpu_ms = (time_ns() - cpu_start) ÷ 1_000_000
    println("Host time: $cpu_ms ms")
    println()

    println("Check difference between final wavefields computed in device and host")
    error = within_epsilon(next_base, next_cpu, nrows, ncols, HALF_LENGTH, 0.1f0)
    println(error ? "FAIL" : "PASS")
    open("wavefield_snapshot_cpu.bin", "w") do io
        write(io, next_cpu)
    end
    println("Final wavefields (from device and CPU) written to disk")
    println("Finished.  ")
    return error ? 1 : 0
end

exit(main(ARGS))
