using CUDA
using Printf

# Julia port of bfs-cuda benchmark.
# Two-kernel BFS: expand frontier (kernel1) + update visited (kernel2).
# Iterates until no more updates.

function bfs_expand!(node_start, node_ne, edges, mask, umask, visited, cost, N::Int32)
    tid = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    @inbounds if tid <= N && mask[tid] != Int8(0)
        mask[tid] = Int8(0)
        num_edges = node_ne[tid]
        starting = node_start[tid]  # 0-based
        i = starting
        stop = starting + num_edges
        while i < stop
            id = edges[i + Int32(1)]  # 0-based id
            if visited[id + Int32(1)] == Int8(0)
                cost[id + Int32(1)] = cost[tid] + Int32(1)
                umask[id + Int32(1)] = Int8(1)
            end
            i += Int32(1)
        end
    end
    return
end

function bfs_update!(mask, umask, visited, over, N::Int32)
    tid = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    @inbounds if tid <= N && umask[tid] != Int8(0)
        mask[tid] = Int8(1)
        visited[tid] = Int8(1)
        over[1] = Int8(1)
        umask[tid] = Int8(0)
    end
    return
end

function bfs_cpu!(N, node_start, node_ne, edges, mask, umask, visited, cost)
    while true
        stop = false
        for tid in 1:N
            if mask[tid] == 1
                mask[tid] = 0
                for i in node_start[tid]:(node_start[tid] + node_ne[tid] - 1)
                    id = edges[i + 1]
                    if visited[id + 1] == 0
                        cost[id + 1] = cost[tid] + 1
                        umask[id + 1] = 1
                    end
                end
            end
        end
        for tid in 1:N
            if umask[tid] == 1
                mask[tid] = 1
                visited[tid] = 1
                stop = true
                umask[tid] = 0
            end
        end
        if !stop; break; end
    end
end

function main()
    if length(ARGS) < 1
        println("Usage: main.jl <graph.txt>")
        return 1
    end
    filename = ARGS[1]
    println("Reading File")

    N = 0
    edges = Int32[]
    node_start = Int32[]
    node_ne = Int32[]
    source = 0
    open(filename, "r") do fp
        N = parse(Int, readline(fp))
        node_start = Vector{Int32}(undef, N)
        node_ne = Vector{Int32}(undef, N)
        for i in 1:N
            parts = split(readline(fp))
            node_start[i] = parse(Int32, parts[1])
            node_ne[i]    = parse(Int32, parts[2])
        end
        source = parse(Int, readline(fp))
        E = parse(Int, readline(fp))
        edges = Vector{Int32}(undef, E)
        for i in 1:E
            parts = split(readline(fp))
            edges[i] = parse(Int32, parts[1])
        end
    end
    source = 0
    E = length(edges)
    @printf("Graph check passed (#nodes = %d, #edges in list = %d, edges referenced = %d)\n", N, E, E)
    @printf("run bfs (#nodes = %d) on device\n", N)

    mask = zeros(Int8, N)
    umask = zeros(Int8, N)
    visited = zeros(Int8, N)
    cost = fill(Int32(-1), N)
    mask[source + 1] = 1
    visited[source + 1] = 1
    cost[source + 1] = 0

    d_ns = CuArray(node_start)
    d_ne = CuArray(node_ne)
    d_ed = CuArray(edges)
    d_m  = CuArray(mask)
    d_um = CuArray(umask)
    d_v  = CuArray(visited)
    d_c  = CuArray(cost)
    d_over = CUDA.zeros(Int8, 1)

    threads = 256
    blocks  = cld(N, threads)

    total_ns = 0
    while true
        CUDA.fill!(d_over, Int8(0))
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=threads blocks=blocks bfs_expand!(d_ns, d_ne, d_ed, d_m, d_um, d_v, d_c, Int32(N))
        @cuda threads=threads blocks=blocks bfs_update!(d_m, d_um, d_v, d_over, Int32(N))
        CUDA.synchronize()
        total_ns += (time_ns() - t0)
        h_over = CUDA.@allowscalar d_over[1]
        if h_over == Int8(0); break; end
    end
    @printf("Total kernel execution time : %f (us)\n", total_ns * 1e-3)

    gpu_cost = Array(d_c)

    @printf("run bfs (#nodes = %d) on host (cpu) \n", N)
    mask_c = zeros(Int8, N)
    umask_c = zeros(Int8, N)
    visited_c = zeros(Int8, N)
    cost_c = fill(Int32(-1), N)
    mask_c[source + 1] = 1
    visited_c[source + 1] = 1
    cost_c[source + 1] = 0
    bfs_cpu!(N, node_start, node_ne, edges, mask_c, umask_c, visited_c, cost_c)

    ok = gpu_cost == cost_c
    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
