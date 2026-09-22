using CUDA
using Printf
using Random

function load_thomas_matrix_syn(size::Int)
    rng = MersenneTwister(1)
    u = [rand(rng) * 4.0 - 2.0 for _ in 1:size]
    l = [rand(rng) * 4.0 - 2.0 for _ in 1:size]
    d = [rand(rng) * 5.0 + 5.0 for _ in 1:size]
    rhs = [rand(rng) * 4.0 - 2.0 for _ in 1:size]
    return u, l, d, rhs
end

function solve_seq!(l, d, u, rhs, n::Int, systems::Int)
    for j0 in 0:(systems - 1)
        first = j0 * n + 1
        last = first + n - 1
        u[first] /= d[first]
        rhs[first] /= d[first]
        for i in (first + 1):(last - 1)
            denom = d[i] - l[i] * u[i - 1]
            u[i] /= denom
            rhs[i] = (rhs[i] - l[i] * rhs[i - 1]) / denom
        end
        rhs[last] = (rhs[last] - l[last] * rhs[last - 1]) / (d[last] - l[last] * u[last - 1])
        for i in (last - 1):-1:first
            rhs[i] -= u[i] * rhs[i + 1]
        end
    end
    return rhs
end

function thomas_kernel!(l, d, u, rhs, m::Int32, batchcount::Int32)
    tid0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if tid0 < batchcount
        first = tid0
        last = batchcount * (m - Int32(1)) + tid0
        fi = Int(first) + 1
        u[fi] /= d[fi]
        rhs[fi] /= d[fi]
        i = first + batchcount
        while i < last
            idx = Int(i) + 1
            prev = Int(i - batchcount) + 1
            denom = d[idx] - l[idx] * u[prev]
            u[idx] /= denom
            rhs[idx] = (rhs[idx] - l[idx] * rhs[prev]) / denom
            i += batchcount
        end
        li = Int(last) + 1
        lp = Int(last - batchcount) + 1
        rhs[li] = (rhs[li] - l[li] * rhs[lp]) / (d[li] - l[li] * u[lp])
        i = last - batchcount
        while i >= first
            idx = Int(i) + 1
            rhs[idx] -= u[idx] * rhs[Int(i + batchcount) + 1]
            i -= batchcount
        end
    end
    return
end

function calc_error(src, dst)
    err = 0.0
    @inbounds for i in eachindex(src, dst)
        err = max(err, abs(abs(src[i]) - abs(dst[i])))
    end
    return err
end

function main()
    length(ARGS) == 4 || begin
        println("Usage: main.jl [system size] [#systems] [thread block size] [repeat]")
        exit(1)
    end
    m = parse(Int, ARGS[1])
    systems = parse(Int, ARGS[2])
    block_size = parse(Int, ARGS[3])
    repeat = parse(Int, ARGS[4])
    m > 1 && systems > 0 && block_size > 0 && repeat > 0 || error("all numeric arguments must be positive")
    CUDA.allowscalar(false)

    base_u, base_l, base_d, base_rhs = load_thomas_matrix_syn(m)
    matrix_size = m * systems
    u_seq = Vector{Float64}(undef, matrix_size)
    d_seq = Vector{Float64}(undef, matrix_size)
    l_seq = Vector{Float64}(undef, matrix_size)
    rhs_seq = Vector{Float64}(undef, matrix_size)
    for sys0 in 0:(systems - 1), j in 1:m
        idx = sys0 * m + j
        u_seq[idx] = base_u[j]
        d_seq[idx] = base_d[j]
        l_seq[idx] = base_l[j]
        rhs_seq[idx] = base_rhs[j]
    end

    start = time_ns()
    for _ in 1:repeat
        solve_seq!(l_seq, d_seq, u_seq, rhs_seq, m, systems)
    end
    elapsed = time_ns() - start
    @printf("Average serial execution time: %f (ms)\n", elapsed * 1.0e-6 / repeat)
    rhs_seq_output = copy(rhs_seq)

    u_input = Vector{Float64}(undef, matrix_size)
    d_input = Vector{Float64}(undef, matrix_size)
    l_input = Vector{Float64}(undef, matrix_size)
    rhs_input = Vector{Float64}(undef, matrix_size)
    for sys0 in 0:(systems - 1), j in 1:m
        idx = sys0 * m + j
        u_input[idx] = base_u[j]
        d_input[idx] = base_d[j]
        l_input[idx] = base_l[j]
        rhs_input[idx] = base_rhs[j]
    end

    u_host = Vector{Float64}(undef, matrix_size)
    d_host = Vector{Float64}(undef, matrix_size)
    l_host = Vector{Float64}(undef, matrix_size)
    rhs_host = Vector{Float64}(undef, matrix_size)
    rhs_seq_interleave = Vector{Float64}(undef, matrix_size)
    for i0 in 0:(m - 1), j0 in 0:(systems - 1)
        dst = i0 * systems + j0 + 1
        src = j0 * m + i0 + 1
        u_host[dst] = u_input[src]
        l_host[dst] = l_input[src]
        d_host[dst] = d_input[src]
        rhs_host[dst] = rhs_input[src]
        rhs_seq_interleave[dst] = rhs_seq_output[src]
    end

    d_u = CuArray(u_host)
    d_l = CuArray(l_host)
    d_d = CuArray(d_host)
    d_rhs = CuArray(rhs_host)
    blocks = cld(systems, block_size)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=blocks thomas_kernel!(d_l, d_d, d_u, d_rhs, Int32(m), Int32(systems))
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (ms)\n", elapsed * 1.0e-6 / repeat)

    rhs_host = Array(d_rhs)
    err = calc_error(rhs_seq_interleave, rhs_host)
    @printf("Maximum error: %e\n", err)
    println(err <= 1.0e-9 ? "PASS" : "FAIL")
end

main()
