using CUDA
using Printf
using Random

function implicit_fma_kernel!(a, b, c, in_indices, out_indices, num_ops::Int32, channels::Int32)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    stride = gridDim().x * blockDim().x
    total = num_ops * channels
    while idx0 < total
        op = idx0 ÷ channels
        ch = idx0 - op * channels
        in_row = in_indices[op + Int32(1)]
        out_row = out_indices[op + Int32(1)]
        src = in_row * channels + ch + Int32(1)
        dst = out_row * channels + ch + Int32(1)
        @inbounds c[dst] += a[src] * b[ch + Int32(1)]
        idx0 += stride
    end
    return
end

function reference!(a::Vector{T}, b::Vector{T}, c::Vector{T},
                    in_indices::Vector{Int32}, out_indices::Vector{Int32},
                    num_ops::Int, channels::Int) where {T}
    for op in 1:num_ops
        in_base = Int(in_indices[op]) * channels
        out_base = Int(out_indices[op]) * channels
        for ch in 1:channels
            c[out_base + ch] += a[in_base + ch] * b[ch]
        end
    end
    return c
end

function fill_inputs(::Type{T}, na::Int, channels::Int, num_ops::Int) where {T}
    rng = MersenneTwister(123)
    a = T.(rand(rng, 0:4, na * channels))
    b = T.(rand(rng, -1:1, channels))
    in_indices = Int32.(rand(rng, 0:(na - 1), num_ops))
    out_indices = Int32.(0:(num_ops - 1))
    return a, b, in_indices, out_indices
end

function run_kernel!(d_a, d_b, d_c, d_in, d_out, num_ops::Int, channels::Int)
    threads = 256
    blocks = cld(num_ops * channels, threads)
    @cuda threads=threads blocks=blocks implicit_fma_kernel!(
        d_a, d_b, d_c, d_in, d_out, Int32(num_ops), Int32(channels))
    return
end

function fma_trial(label::String, ::Type{T}, na::Int, nc::Int, channels::Int,
                   num_ops::Int, repeat::Int) where {T}
    println(label)
    a, b, in_indices, out_indices = fill_inputs(T, na, channels, num_ops)
    c_size = nc * channels

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CUDA.zeros(T, c_size)
    d_in = CuArray(in_indices)
    d_out = CuArray(out_indices)

    run_kernel!(d_a, d_b, d_c, d_in, d_out, num_ops, channels)
    CUDA.synchronize()
    h_basic = Array(d_c)

    ref = zeros(T, c_size)
    reference!(a, b, ref, in_indices, out_indices, num_ops, channels)
    ok = all(abs(Float64(h_basic[i]) - Float64(ref[i])) <= 1.0e-3 for i in eachindex(ref))

    CUDA.fill!(d_c, zero(T))
    run_kernel!(d_a, d_b, d_c, d_in, d_out, num_ops, channels)
    CUDA.synchronize()
    h_rowwise = Array(d_c)
    ok &= all(abs(Float64(h_rowwise[i]) - Float64(ref[i])) <= 1.0e-3 for i in eachindex(ref))
    println(ok ? "PASS" : "FAIL")

    CUDA.fill!(d_c, zero(T))
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        CUDA.fill!(d_c, zero(T))
        run_kernel!(d_a, d_b, d_c, d_in, d_out, num_ops, channels)
    end
    CUDA.synchronize()
    @printf("Average execution time of basic kernel: %f (us)\n",
            (time_ns() - start) * 1.0e-3 / repeat)

    CUDA.fill!(d_c, zero(T))
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        CUDA.fill!(d_c, zero(T))
        run_kernel!(d_a, d_b, d_c, d_in, d_out, num_ops, channels)
    end
    CUDA.synchronize()
    @printf("Average execution time of rowwise kernel: %f (us)\n",
            (time_ns() - start) * 1.0e-3 / repeat)
end

function main(args)
    if length(args) != 5
        println("Usage: main.jl <NA> <NC> <C> <num_ops> <repeat>")
        println("<Number of rows in A> <Number of rows in C> <Number of channels/columns> <Number of operations>")
        return 1
    end

    na = parse(Int, args[1])
    nc = parse(Int, args[2])
    channels = parse(Int, args[3])
    num_ops = parse(Int, args[4])
    repeat = parse(Int, args[5])
    if num_ops > nc
        println("Error: Number of operations is larger than number of rows in C")
        return 1
    end

    for _ in 1:2
        fma_trial("FP16 FMA", Float16, na, nc, channels, num_ops, repeat)
        fma_trial("BF16 FMA", Float32, na, nc, channels, num_ops, repeat)
        fma_trial("FP32 FMA", Float32, na, nc, channels, num_ops, repeat)
        fma_trial("FP64 FMA", Float64, na, nc, channels, num_ops, repeat)
    end
    return 0
end

exit(main(ARGS))
