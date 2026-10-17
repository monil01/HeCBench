using CUDA
using Printf
using Random
using StaticArrays

const THREADS = 512
const RADIUS = Int32(512)

@inline function init_dist8()
    return MVector{8, Int32}(RADIUS, RADIUS, RADIUS, RADIUS, RADIUS, RADIUS, RADIUS, RADIUS)
end

@inline function init_dist16()
    return MVector{16, Int32}(RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS)
end

@inline function init_dist32()
    return MVector{32, Int32}(RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS,
                              RADIUS, RADIUS, RADIUS, RADIUS)
end

function cube_select_kernel!(n::Int32, radius::Int32, xyz, idx_out)
    batch_idx = blockIdx().x - Int32(1)
    base_xyz = batch_idx * n * Int32(3)
    base_out = batch_idx * n * Int32(8)
    i = threadIdx().x - Int32(1)
    while i < n
        x = @inbounds xyz[base_xyz + i * Int32(3) + Int32(1)]
        y = @inbounds xyz[base_xyz + i * Int32(3) + Int32(2)]
        z = @inbounds xyz[base_xyz + i * Int32(3) + Int32(3)]
        temp_dist = init_dist8()
        for k in Int32(0):Int32(7)
            @inbounds idx_out[base_out + i * Int32(8) + k + Int32(1)] = i
        end
        for j in Int32(0):(n - Int32(1))
            j == i && continue
            tx = @inbounds xyz[base_xyz + j * Int32(3) + Int32(1)]
            ty = @inbounds xyz[base_xyz + j * Int32(3) + Int32(2)]
            tz = @inbounds xyz[base_xyz + j * Int32(3) + Int32(3)]
            dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
            dist > radius && continue
            temp_idx = (tx > x ? Int32(4) : Int32(0)) + (ty > y ? Int32(2) : Int32(0)) + (tz > z ? Int32(1) : Int32(0))
            slot = Int(temp_idx + Int32(1))
            if dist < temp_dist[slot]
                @inbounds idx_out[base_out + i * Int32(8) + temp_idx + Int32(1)] = j
                temp_dist[slot] = dist
            end
        end
        i += blockDim().x
    end
    return
end

function cube_select_two_kernel!(n::Int32, radius::Int32, xyz, idx_out)
    batch_idx = blockIdx().x - Int32(1)
    base_xyz = batch_idx * n * Int32(3)
    base_out = batch_idx * n * Int32(16)
    i = threadIdx().x - Int32(1)
    while i < n
        x = @inbounds xyz[base_xyz + i * Int32(3) + Int32(1)]
        y = @inbounds xyz[base_xyz + i * Int32(3) + Int32(2)]
        z = @inbounds xyz[base_xyz + i * Int32(3) + Int32(3)]
        temp_dist = init_dist16()
        for k in Int32(0):Int32(15)
            @inbounds idx_out[base_out + i * Int32(16) + k + Int32(1)] = i
        end
        for j in Int32(0):(n - Int32(1))
            j == i && continue
            tx = @inbounds xyz[base_xyz + j * Int32(3) + Int32(1)]
            ty = @inbounds xyz[base_xyz + j * Int32(3) + Int32(2)]
            tz = @inbounds xyz[base_xyz + j * Int32(3) + Int32(3)]
            dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
            dist > radius && continue
            temp_idx = (tx > x ? Int32(8) : Int32(0)) + (ty > y ? Int32(4) : Int32(0)) + (tz > z ? Int32(2) : Int32(0))
            flag = false
            for k in Int32(0):Int32(1)
                slot = Int(temp_idx + k + Int32(1))
                if dist < temp_dist[slot]
                    flag = true
                end
                if flag
                    for kk in Int32(1):-Int32(1):(k + Int32(1))
                        @inbounds idx_out[base_out + i * Int32(16) + temp_idx + kk + Int32(1)] =
                            idx_out[base_out + i * Int32(16) + temp_idx + kk]
                        temp_dist[Int(temp_idx + kk + Int32(1))] = temp_dist[Int(temp_idx + kk)]
                    end
                    @inbounds idx_out[base_out + i * Int32(16) + temp_idx + k + Int32(1)] = j
                    temp_dist[slot] = dist
                    break
                end
            end
        end
        i += blockDim().x
    end
    return
end

function cube_select_four_kernel!(n::Int32, radius::Int32, xyz, idx_out)
    batch_idx = blockIdx().x - Int32(1)
    base_xyz = batch_idx * n * Int32(3)
    base_out = batch_idx * n * Int32(32)
    i = threadIdx().x - Int32(1)
    while i < n
        x = @inbounds xyz[base_xyz + i * Int32(3) + Int32(1)]
        y = @inbounds xyz[base_xyz + i * Int32(3) + Int32(2)]
        z = @inbounds xyz[base_xyz + i * Int32(3) + Int32(3)]
        temp_dist = init_dist32()
        for k in Int32(0):Int32(31)
            @inbounds idx_out[base_out + i * Int32(32) + k + Int32(1)] = i
        end
        for j in Int32(0):(n - Int32(1))
            j == i && continue
            tx = @inbounds xyz[base_xyz + j * Int32(3) + Int32(1)]
            ty = @inbounds xyz[base_xyz + j * Int32(3) + Int32(2)]
            tz = @inbounds xyz[base_xyz + j * Int32(3) + Int32(3)]
            dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
            dist > radius && continue
            temp_idx = (tx > x ? Int32(16) : Int32(0)) + (ty > y ? Int32(8) : Int32(0)) + (tz > z ? Int32(4) : Int32(0))
            flag = false
            for k in Int32(0):Int32(3)
                slot = Int(temp_idx + k + Int32(1))
                if dist < temp_dist[slot]
                    flag = true
                end
                if flag
                    for kk in Int32(3):-Int32(1):(k + Int32(1))
                        @inbounds idx_out[base_out + i * Int32(32) + temp_idx + kk + Int32(1)] =
                            idx_out[base_out + i * Int32(32) + temp_idx + kk]
                        temp_dist[Int(temp_idx + kk + Int32(1))] = temp_dist[Int(temp_idx + kk)]
                    end
                    @inbounds idx_out[base_out + i * Int32(32) + temp_idx + k + Int32(1)] = j
                    temp_dist[slot] = dist
                    break
                end
            end
        end
        i += blockDim().x
    end
    return
end

function cube_select_cpu(b, n, radius, xyz, width)
    out = Vector{Int32}(undef, b * n * width)
    ranks = width == 8 ? 1 : width == 16 ? 2 : 4
    for batch_idx in 0:(b - 1)
        base_xyz = batch_idx * n * 3
        base_out = batch_idx * n * width
        for i in 0:(n - 1)
            x = xyz[base_xyz + i * 3 + 1]
            y = xyz[base_xyz + i * 3 + 2]
            z = xyz[base_xyz + i * 3 + 3]
            temp_dist = fill(Int32(radius), width)
            for k in 0:(width - 1)
                out[base_out + i * width + k + 1] = Int32(i)
            end
            for j in 0:(n - 1)
                j == i && continue
                tx = xyz[base_xyz + j * 3 + 1]
                ty = xyz[base_xyz + j * 3 + 2]
                tz = xyz[base_xyz + j * 3 + 3]
                dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
                dist > radius && continue
                temp_idx = ((tx > x) ? 4 * ranks : 0) + ((ty > y) ? 2 * ranks : 0) + ((tz > z) ? ranks : 0)
                for k in 0:(ranks - 1)
                    if dist < temp_dist[temp_idx + k + 1]
                        for kk in (ranks - 1):-1:(k + 1)
                            out[base_out + i * width + temp_idx + kk + 1] = out[base_out + i * width + temp_idx + kk]
                            temp_dist[temp_idx + kk + 1] = temp_dist[temp_idx + kk]
                        end
                        out[base_out + i * width + temp_idx + k + 1] = Int32(j)
                        temp_dist[temp_idx + k + 1] = dist
                        break
                    end
                end
            end
        end
    end
    return out
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of batches> <number of points> <repeat>")
        return 1
    end
    b = parse(Int, args[1])
    n = parse(Int, args[2])
    repeat = parse(Int, args[3])

    rng = MersenneTwister(123)
    h_xyz = Int32.(rand(rng, -256:255, b * n * 3))
    d_xyz = CuArray(h_xyz)
    d_out = CUDA.zeros(Int32, b * n * 8)
    d_out2 = CUDA.zeros(Int32, b * n * 16)
    d_out4 = CUDA.zeros(Int32, b * n * 32)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=b cube_select_kernel!(Int32(n), RADIUS, d_xyz, d_out)
    end
    CUDA.synchronize()
    @printf("Average execution time of select kernel: %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
    error = Array(d_out) == cube_select_cpu(b, n, Int(RADIUS), h_xyz, 8) ? 0 : 1

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=b cube_select_two_kernel!(Int32(n), RADIUS, d_xyz, d_out2)
    end
    CUDA.synchronize()
    @printf("Average execution time of select2 kernel: %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
    error += Array(d_out2) == cube_select_cpu(b, n, Int(RADIUS), h_xyz, 16) ? 0 : 1

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=THREADS blocks=b cube_select_four_kernel!(Int32(n), RADIUS, d_xyz, d_out4)
    end
    CUDA.synchronize()
    @printf("Average execution time of select4 kernel: %f (us)\n", (time_ns() - t0) * 1.0e-3 / repeat)
    error += Array(d_out4) == cube_select_cpu(b, n, Int(RADIUS), h_xyz, 32) ? 0 : 1

    println(error == 0 ? "PASS" : "FAIL")
    return error == 0 ? 0 : 1
end

exit(main(ARGS))
