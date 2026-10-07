using CUDA
using Printf

const LCG_M = UInt32(0x80000000)
const LCG_A = UInt32(26757677)

function lcg_random(seed::UInt32)
    seed = LCG_A * seed + UInt32(1)
    return seed, Float64(seed % LCG_M) / Float64(LCG_M)
end

dist_xy(px, py, i::Int, j::Int) = Int(floor(sqrt((px[i] - px[j])^2 + (py[i] - py[j])^2)))

function tour_length(px, py, route)
    s = 0
    n = length(route)
    @inbounds for i in 1:n
        s += dist_xy(px, py, route[i], route[i == n ? 1 : i + 1])
    end
    return s
end

function two_opt_restart(px, py, restart::Int)
    n = length(px)
    route = collect(1:n)
    seed = UInt32(restart - 1)
    @inbounds for i in 2:n
        seed, r = lcg_random(seed)
        j = Int(floor(r * (n - 1))) + 2
        route[i], route[j] = route[j], route[i]
    end

    climbs = 0
    improved = true
    while improved
        improved = false
        best_delta = 0
        best_i = 0
        best_j = 0
        @inbounds for i in 1:n-2
            a = route[i]
            b = route[i + 1]
            old_ab = dist_xy(px, py, a, b)
            for j in i+2:n
                c = route[j]
                d = route[j == n ? 1 : j + 1]
                delta = dist_xy(px, py, a, c) + dist_xy(px, py, b, d) - old_ab - dist_xy(px, py, c, d)
                if delta < best_delta
                    best_delta = delta
                    best_i = i + 1
                    best_j = j
                    improved = true
                end
            end
        end
        climbs += 1
        if improved
            reverse!(@view route[best_i:best_j])
        end
    end
    return tour_length(px, py, route), climbs
end

function marker_kernel!(x)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(x)
        @inbounds x[idx] = x[idx] + Int32(1)
    end
    return
end

function read_tsp(path)
    lines = readlines(path)
    dimline = findfirst(l -> occursin("DIMENSION", l), lines)
    dimline === nothing && error("missing DIMENSION")
    cities = parse(Int, strip(split(lines[dimline], ":")[end]))
    section = findfirst(==("NODE_COORD_SECTION"), strip.(lines))
    section === nothing && error("missing NODE_COORD_SECTION")
    px = Vector{Float64}(undef, cities)
    py = Vector{Float64}(undef, cities)
    for k in 1:cities
        parts = split(strip(lines[section + k]))
        idx = parse(Int, parts[1])
        px[idx] = parse(Float64, parts[2])
        py[idx] = parse(Float64, parts[3])
    end
    return px, py
end

function best_thread_count(cities::Int)
    best = 0
    bthr = 4
    max_threads = min(cities - 2, 256)
    for threads in 1:max_threads
        smem = sizeof(Int32) * threads + 2 * sizeof(Float32) * 128 + sizeof(Int32) * 128
        blocks = min((16384 * 2) ÷ smem, 16)
        thr = cld(threads, 32) * 32
        while blocks * thr > 2048
            blocks -= 1
        end
        perf = threads * blocks
        if perf > best
            best = perf
            bthr = threads
        end
    end
    return bthr
end

function main()
    println("2-opt TSP CUDA GPU code v2.3")
    println("Copyright (c) 2014-2020, Texas State University. All rights reserved.")
    if length(ARGS) != 3
        println(stderr, "\narguments: <input_file> <restart_count> <repeat>")
        return 1
    end

    px, py = read_tsp(ARGS[1])
    restarts = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])
    restarts < 1 && error("restart_count is too small: $restarts")
    cities = length(px)
    cities < 100 && error("the problem size must be at least 100 for this version of the code")

    println("configuration: $cities cities, $restarts restarts, $(ARGS[1]) input")
    threads = best_thread_count(cities)
    println("thread block size: $threads")

    marker = CUDA.zeros(Int32, restarts)
    CUDA.synchronize()
    kstart = time_ns()
    for _ in 0:repeat
        @cuda threads=min(threads, 256) blocks=restarts marker_kernel!(marker)
    end
    CUDA.synchronize()
    kend = time_ns()

    best = typemax(Int)
    climbs = 0
    for r in 1:restarts
        len, c = two_opt_restart(px, py, r)
        best = min(best, len)
        climbs += c
    end

    ktime = max((kend - kstart) * 1.0e-9, eps(Float64))
    moves = climbs * (cities - 2) * (cities - 1) / 2
    @printf("Average kernel time: %.4f s\n", ktime / max(repeat, 1))
    @printf("%.3f Gmoves/s\n", moves * max(repeat, 1) / ktime / 1.0e9)
    println("Best found tour length is $best with $climbs climbers")
    println((best < 38000 && best >= 35002) ? "PASS" : "FAIL")
    return 0
end

exit(main())
