using CUDA
using Printf

struct XYZ
    x::Float32
    y::Float32
    z::Float32
end

function parse_args(args)
    work_group_size = 256
    file_name = "../bezier-surface-cuda/input/control.txt"
    in_size = 3
    out_size = 300

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h"
            println("Usage: main.jl [-g work_group_size] [-f input_file] [-m input_size] [-n output_size]")
            return nothing
        elseif arg == "-g" && i < length(args)
            i += 1
            work_group_size = parse(Int, args[i])
        elseif arg == "-f" && i < length(args)
            i += 1
            file_name = args[i]
        elseif arg == "-m" && i < length(args)
            i += 1
            in_size = parse(Int, args[i])
        elseif arg == "-n" && i < length(args)
            i += 1
            out_size = parse(Int, args[i])
        else
            error("Unrecognized option: $arg")
        end
        i += 1
    end

    return (; work_group_size, file_name, in_size_i=in_size, in_size_j=in_size,
            out_size_i=out_size, out_size_j=out_size)
end

function read_input(file_name::AbstractString, in_size_i::Int, in_size_j::Int)
    open(file_name, "r") do io
        println("Read data from file $file_name")
        points = XYZ[]
        for line in eachline(io)
            fields = split(strip(line), ',')
            length(fields) == 3 || continue
            push!(points, XYZ(parse(Float32, fields[1]),
                              parse(Float32, fields[2]),
                              parse(Float32, fields[3])))
        end

        isempty(points) && error("input file has no control points")
        out = Vector{XYZ}(undef, (in_size_i + 1) * (in_size_j + 1))
        k = 1
        for i in 0:in_size_i, j in 0:in_size_j
            out[i * (in_size_j + 1) + j + 1] = points[k]
            k = k == 16 ? 1 : k + 1
        end
        return out
    end
end

function bezier_blend(k::Int, mu::Float32, n::Int)
    nn = n
    kn = k
    nkn = n - k
    blend = Float32(1)
    while nn >= 1
        blend *= Float32(nn)
        nn -= 1
        if kn > 1
            blend /= Float32(kn)
            kn -= 1
        end
        if nkn > 1
            blend /= Float32(nkn)
            nkn -= 1
        end
    end
    if k > 0
        blend *= mu ^ Float32(k)
    end
    if n - k > 0
        blend *= (Float32(1) - mu) ^ Float32(n - k)
    end
    return blend
end

function bezier_cpu(inp::Vector{XYZ}, ni::Int, nj::Int, resolution_i::Int, resolution_j::Int)
    outp = Vector{XYZ}(undef, resolution_i * resolution_j)
    for i in 0:(resolution_i - 1)
        mui = Float32(i) / Float32(resolution_i - 1)
        for j in 0:(resolution_j - 1)
            muj = Float32(j) / Float32(resolution_j - 1)
            ox = Float32(0)
            oy = Float32(0)
            oz = Float32(0)
            for ki in 0:ni
                bi = bezier_blend(ki, mui, ni)
                for kj in 0:nj
                    bj = bezier_blend(kj, muj, nj)
                    p = inp[ki * (nj + 1) + kj + 1]
                    w = bi * bj
                    ox += p.x * w
                    oy += p.y * w
                    oz += p.z * w
                end
            end
            outp[i * resolution_j + j + 1] = XYZ(ox, oy, oz)
        end
    end
    return outp
end

function bezier_gpu_kernel(inp, outp, ni::Int32, nj::Int32, resolution_i::Int32, resolution_j::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i >= resolution_i
        return
    end

    mui = Float32(i) / Float32(resolution_i - Int32(1))
    for j in Int32(0):(resolution_j - Int32(1))
        muj = Float32(j) / Float32(resolution_j - Int32(1))
        ox = Float32(0)
        oy = Float32(0)
        oz = Float32(0)
        for ki in Int32(0):ni
            bi = bezier_blend(Int(ki), mui, Int(ni))
            for kj in Int32(0):nj
                bj = bezier_blend(Int(kj), muj, Int(nj))
                p = @inbounds inp[ki * (nj + Int32(1)) + kj + Int32(1)]
                w = bi * bj
                ox += p.x * w
                oy += p.y * w
                oz += p.z * w
            end
        end
        @inbounds outp[i * resolution_j + j + Int32(1)] = XYZ(ox, oy, oz)
    end
    return
end

function compare_output(gpu_out::Vector{XYZ}, cpu_out::Vector{XYZ})
    sum_delta = 0.0
    sum_ref = 0.0
    @inbounds for idx in eachindex(cpu_out)
        g = gpu_out[idx]
        c = cpu_out[idx]
        sum_delta += abs(Float64(g.x - c.x)) + abs(Float64(g.y - c.y)) + abs(Float64(g.z - c.z))
        sum_ref += abs(Float64(c.x)) + abs(Float64(c.y)) + abs(Float64(c.z))
    end
    l1norm = sum_delta / sum_ref
    if l1norm >= 1e-6
        println("Test failed")
        return 1
    end
    return 0
end

function main(args)
    params = parse_args(args)
    params === nothing && return 0

    input = read_input(params.file_name, params.in_size_i, params.in_size_j)

    cpu_start = time_ns()
    cpu_out = bezier_cpu(input, params.in_size_i, params.in_size_j,
                         params.out_size_i, params.out_size_j)
    cpu_ms = (time_ns() - cpu_start) ÷ 1_000_000
    println("host execution time: $cpu_ms ms")

    d_in = CuArray(input)
    d_out = CuArray{XYZ}(undef, params.out_size_i * params.out_size_j)
    threads = params.work_group_size
    blocks = cld(params.out_size_i, threads)

    CUDA.synchronize()
    kernel_start = time_ns()
    @cuda threads=threads blocks=blocks bezier_gpu_kernel(
        d_in, d_out, Int32(params.in_size_i), Int32(params.in_size_j),
        Int32(params.out_size_i), Int32(params.out_size_j))
    CUDA.synchronize()
    kernel_ms = (time_ns() - kernel_start) ÷ 1_000_000
    println("kernel execution time: $kernel_ms ms")

    gpu_out = Array(d_out)
    status = compare_output(gpu_out, cpu_out)
    println(status == 0 ? "PASS" : "FAIL")
    return status
end

exit(main(ARGS))
