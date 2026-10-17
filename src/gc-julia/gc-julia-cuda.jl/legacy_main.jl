using CUDA
using Printf

const THREADS_PER_BLOCK = 256

struct ECLGraph
    nodes::Int32
    edges::Int32
    nindex::Vector{Int32}
    nlist::Vector{Int32}
end

function read_ecl_graph(path::String)
    open(path, "r") do io
        nodes = read(io, Int32)
        edges = read(io, Int32)
        nodes >= 1 || error("failed to read valid nodes")
        edges >= 0 || error("failed to read valid edges")
        nindex = Vector{Int32}(undef, Int(nodes) + 1)
        read!(io, nindex)
        nlist = Vector{Int32}(undef, Int(edges))
        read!(io, nlist)
        nindex[1] == 0 || error("neighbor index list always starts at value 0")
        nindex[end] == edges || error("final neighbor index does not equal edge count")
        return ECLGraph(nodes, edges, nindex, nlist)
    end
end

function color_kernel!(nidx, nlist, color, changed, nodes::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = gridDim().x * blockDim().x
    v = tid0
    while v < nodes
        idx = v + Int32(1)
        if color[idx] < Int32(0)
            used = UInt64(0)
            beg = nidx[idx]
            stop = nidx[idx + Int32(1)]
            i = beg
            while i < stop
                nei = nlist[i + Int32(1)] + Int32(1)
                c = color[nei]
                if c >= Int32(0) && c < Int32(64)
                    used |= UInt64(1) << UInt64(c)
                end
                i += Int32(1)
            end

            chosen = Int32(0)
            while chosen < Int32(63) && ((used >> UInt64(chosen)) & UInt64(1)) != UInt64(0)
                chosen += Int32(1)
            end
            color[idx] = chosen
            changed[1] = Int32(1)
        end
        v += stride
    end
    return
end

function repair_conflicts_kernel!(nidx, nlist, color, changed, nodes::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = gridDim().x * blockDim().x
    v = tid0
    while v < nodes
        idx = v + Int32(1)
        c = color[idx]
        if c >= Int32(0)
            beg = nidx[idx]
            stop = nidx[idx + Int32(1)]
            i = beg
            while i < stop
                nei0 = nlist[i + Int32(1)]
                if v > nei0 && color[nei0 + Int32(1)] == c
                    color[idx] = Int32(-1)
                    changed[1] = Int32(1)
                    break
                end
                i += Int32(1)
            end
        end
        v += stride
    end
    return
end

function compute_colors(g::ECLGraph, repeat::Int)
    dev = CUDA.device()
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    mtpsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    blocks = max(1, Int(sms * mtpsm ÷ THREADS_PER_BLOCK))

    @printf("Total number of compute units: %d\n", sms)
    @printf("Maximum resident threads per compute unit: %d\n", mtpsm)
    @printf("Work-group size: %d\n", THREADS_PER_BLOCK)
    @printf("Total number of work-groups: %d\n", blocks)

    d_nidx = CuArray(g.nindex)
    d_nlist = CuArray(g.nlist)
    d_color = CUDA.fill(Int32(-1), Int(g.nodes))
    d_changed = CUDA.zeros(Int32, 1)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        CUDA.fill!(d_color, Int32(-1))
        iter = 0
        while true
            iter += 1
            CUDA.fill!(d_changed, Int32(0))
            @cuda blocks=blocks threads=THREADS_PER_BLOCK color_kernel!(
                d_nidx, d_nlist, d_color, d_changed, g.nodes)
            @cuda blocks=blocks threads=THREADS_PER_BLOCK repair_conflicts_kernel!(
                d_nidx, d_nlist, d_color, d_changed, g.nodes)
            CUDA.synchronize()
            changed = CUDA.@allowscalar d_changed[1]
            if changed == 0 || iter >= 1024
                break
            end
        end
    end
    CUDA.synchronize()
    runtime = ((time_ns() - start) * 1e-9) / repeat

    @printf("average runtime (%d runs):    %.6f s\n", repeat, runtime)
    @printf("throughput: %.6f Mnodes/s\n", Float64(g.nodes) * 0.000001 / runtime)
    @printf("throughput: %.6f Medges/s\n", Float64(g.edges) * 0.000001 / runtime)
    return Array(d_color)
end

function verify_and_print(g::ECLGraph, color::Vector{Int32})
    ok = true
    for v0 in 0:Int(g.nodes)-1
        v = v0 + 1
        if color[v] < 0
            @printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n",
                    v0, g.nindex[v + 1] - g.nindex[v])
            ok = false
            break
        end
        for i0 in g.nindex[v]:(g.nindex[v + 1] - Int32(1))
            nei = g.nlist[Int(i0) + 1] + Int32(1)
            if color[Int(nei)] == color[v]
                @printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n",
                        color[v], v0, nei - Int32(1))
                ok = false
                break
            end
        end
        ok || break
    end
    println(ok ? "PASS" : "FAIL")

    if ok
        vals = 16
        counts = zeros(Int, vals)
        cols = maximum(color) + Int32(1)
        for c in color
            if 0 <= c < vals
                counts[Int(c) + 1] += 1
            end
        end
        @printf("Number of distinct colors used: %d\n", cols)
        running = 0
        for i in 0:min(vals, Int(cols))-1
            running += counts[i + 1]
            @printf("color %2d: %10d (%5.1f%%)\n", i, counts[i + 1],
                    100.0 * running / Int(g.nodes))
        end
    end
    return ok
end

function main()
    println("ECL-GC v1.2 (main.cu)")
    println("Copyright 2020 Texas State University\n")
    if length(ARGS) != 2
        @printf("USAGE: %s <input_file_name> <repeat>\n\n", PROGRAM_FILE)
        return 1
    end

    g = read_ecl_graph(ARGS[1])
    repeat = parse(Int, ARGS[2])
    @printf("input: %s\n", ARGS[1])
    @printf("nodes: %d\n", g.nodes)
    @printf("edges: %d\n", g.edges)
    @printf("avg degree: %.2f\n", Float64(g.edges) / Float64(g.nodes))

    color = compute_colors(g, repeat)
    ok = verify_and_print(g, color)
    return ok ? 0 : 1
end

exit(main())
