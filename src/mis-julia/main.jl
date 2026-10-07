using CUDA
using Printf

const THREADS_PER_BLOCK = 256
const IN_STATUS = UInt8(0xfe)
const OUT_STATUS = UInt8(0x00)

struct ECLGraph
    nodes::Int32
    edges::Int32
    nindex::Vector{Int32}
    nlist::Vector{Int32}
end

function read_i32(io)
    return Int32(read(io, Int32))
end

function read_ecl_graph(path::String)
    open(path, "r") do io
        nodes = read_i32(io)
        nodes < 1 && error("failed to read valid nodes")
        edges = read_i32(io)
        edges < 0 && error("failed to read valid edges")

        nindex = Vector{Int32}(undef, Int(nodes) + 1)
        read!(io, nindex)
        nindex[1] != 0 && error("neighbor index list always starts at value 0")
        nindex[end] != edges && error("final neighbor index does not equal edge count")
        for v in 1:Int(nodes)
            nindex[v] > nindex[v + 1] && error("neighbor index list must be non-decreasing")
        end

        nlist = Vector{Int32}(undef, Int(edges))
        read!(io, nlist)
        for dst in nlist
            (dst < 0 || dst >= nodes) && error("neighbor index out of range")
        end

        return ECLGraph(nodes, edges, nindex, nlist)
    end
end

function resolve_input(path::String)
    isfile(path) && return path
    fallback = joinpath(@__DIR__, "..", "mis-cuda", basename(path))
    isfile(fallback) && return fallback
    return path
end

function hash_u32(val::UInt32)
    val = ((val >> 16) ⊻ val) * UInt32(0x045d9f3b)
    val = ((val >> 16) ⊻ val) * UInt32(0x045d9f3b)
    return (val >> 16) ⊻ val
end

function init_kernel!(nodes::Int32, edges::Int32, nidx, nstat)
    from = (blockIdx().x - Int32(1)) * Int32(THREADS_PER_BLOCK) +
           (threadIdx().x - Int32(1))
    incr = gridDim().x * Int32(THREADS_PER_BLOCK)
    avg = Float32(edges) / Float32(nodes)
    scaledavg = Float32((Int32(IN_STATUS) ÷ Int32(2)) - Int32(1)) * avg

    i0 = from
        while i0 < nodes
            idx = i0 + Int32(1)
            val = IN_STATUS
            degree = @inbounds nidx[idx + Int32(1)] - nidx[idx]
        if degree > Int32(0)
            x = Float32(degree) -
                (Float32(hash_u32(UInt32(i0))) * Float32(0.00000000023283064365386962890625))
            res = Int32(scaledavg / (avg + x))
            val = UInt8(UInt32((res + res) | Int32(1)) & UInt32(0xff))
        end
        @inbounds nstat[idx] = val
        i0 += incr
    end
    return
end

function findmins_kernel!(nodes::Int32, nidx, nlist, nstat)
    from = (blockIdx().x - Int32(1)) * Int32(THREADS_PER_BLOCK) +
           (threadIdx().x - Int32(1))
    incr = gridDim().x * Int32(THREADS_PER_BLOCK)

    missing = Int32(0)
    while true
        missing = Int32(0)
        v0 = from
        while v0 < nodes
            vidx = v0 + Int32(1)
            nv = @inbounds nstat[vidx]
            if (nv & UInt8(0x01)) != UInt8(0)
                i = @inbounds nidx[vidx]
                stop = @inbounds nidx[vidx + Int32(1)]
                while i < stop
                    nb0 = @inbounds nlist[i + Int32(1)]
                    nbstat = @inbounds nstat[nb0 + Int32(1)]
                    if !((nv > nbstat) || ((nv == nbstat) && (v0 > nb0)))
                        break
                    end
                    i += Int32(1)
                end
                if i < stop
                    missing = Int32(1)
                else
                    j = @inbounds nidx[vidx]
                    while j < stop
                        nb0 = @inbounds nlist[j + Int32(1)]
                        @inbounds nstat[nb0 + Int32(1)] = OUT_STATUS
                        j += Int32(1)
                    end
                    @inbounds nstat[vidx] = IN_STATUS
                end
            end
            v0 += incr
        end
        missing == Int32(0) && break
    end
    return
end

function compute_mis(repeat::Int32, g::ECLGraph)
    d_nidx = CuArray(g.nindex)
    d_nlist = CuArray(g.nlist)
    d_nstat = CUDA.zeros(UInt8, Int(g.nodes))

    blocks = 24
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:Int(repeat)
        @cuda threads=THREADS_PER_BLOCK blocks=blocks init_kernel!(g.nodes, g.edges, d_nidx, d_nstat)
        @cuda threads=THREADS_PER_BLOCK blocks=blocks findmins_kernel!(g.nodes, d_nidx, d_nlist, d_nstat)
    end
    CUDA.synchronize()
    runtime = ((time_ns() - t0) / 1.0e9) / Float64(repeat)

    @printf("compute time: %.6f s\n", runtime)
    @printf("throughput: %.6f Mnodes/s\n", Float64(g.nodes) * 0.000001 / runtime)
    @printf("throughput: %.6f Medges/s\n", Float64(g.edges) * 0.000001 / runtime)
    @printf("Average kernel execution time : %.6f s\n", runtime)

    return Array(d_nstat)
end

function verify_mis(g::ECLGraph, nstatus::Vector{UInt8})
    ok = true
    for v0 in Int32(0):(g.nodes - Int32(1))
        vidx = Int(v0) + 1
        if nstatus[vidx] != IN_STATUS && nstatus[vidx] != OUT_STATUS
            println(stderr, "ERROR: found unprocessed node in graph\n")
            ok = false
            break
        end

        start = g.nindex[vidx]
        stop = g.nindex[vidx + 1]
        if nstatus[vidx] == IN_STATUS
            for i0 in start:(stop - Int32(1))
                if nstatus[Int(g.nlist[Int(i0) + 1]) + 1] == IN_STATUS
                    println(stderr, "ERROR: found adjacent nodes in MIS\n")
                    ok = false
                    break
                end
            end
        else
            flag = false
            for i0 in start:(stop - Int32(1))
                if nstatus[Int(g.nlist[Int(i0) + 1]) + 1] == IN_STATUS
                    flag = true
                    break
                end
            end
            if !flag
                println(stderr, "ERROR: set is not maximal\n")
                ok = false
            end
        end
        ok || break
    end
    return ok
end

function main()
    println("ECL-MIS v1.3 (main.cu)")
    println("Copyright 2017-2020 Texas State University")

    if length(ARGS) != 2
        println(stderr, "USAGE: main.jl <input_file_name> <repeat>\n")
        return 1
    end

    input_arg = ARGS[1]
    repeat = parse(Int32, ARGS[2])
    g = read_ecl_graph(resolve_input(input_arg))

    @printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, input_arg)
    @printf("average degree: %.2f edges per node\n", Float64(g.edges) / Float64(g.nodes))

    nstatus = compute_mis(repeat, g)
    ok = verify_mis(g, nstatus)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
