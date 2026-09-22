using CUDA
using Printf
using Random

const GPU_THREADS = 256

function sequence_mask_kernel!(n::Int32, m::Int32, b::Int32, input, seq_lengths, fill_val::Int32, output)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = b * n * m
    stride = blockDim().x * gridDim().x
    while idx0 < total
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        @inbounds output[idx0 + Int32(1)] = k >= seq_lengths[j + Int32(1)] ? fill_val : input[idx0 + Int32(1)]
        idx0 += stride
    end
    return
end

function window_mask_kernel!(n::Int32, m::Int32, b::Int32, input, centers, radius::Int32, fill_val::Int32, output)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = b * n * m
    stride = blockDim().x * gridDim().x
    while idx0 < total
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        c = @inbounds centers[j + Int32(1)]
        @inbounds output[idx0 + Int32(1)] = (k < c - radius || k > c + radius) ? fill_val : input[idx0 + Int32(1)]
        idx0 += stride
    end
    return
end

function upper_mask_kernel!(n::Int32, m::Int32, b::Int32, input, fill_val::Int32, output)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = b * n * m
    stride = blockDim().x * gridDim().x
    while idx0 < total
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        @inbounds output[idx0 + Int32(1)] = k > j ? fill_val : input[idx0 + Int32(1)]
        idx0 += stride
    end
    return
end

function lower_mask_kernel!(n::Int32, m::Int32, b::Int32, input, fill_val::Int32, output)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = b * n * m
    stride = blockDim().x * gridDim().x
    while idx0 < total
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        @inbounds output[idx0 + Int32(1)] = k < j ? fill_val : input[idx0 + Int32(1)]
        idx0 += stride
    end
    return
end

function upper_diag_mask_kernel!(n::Int32, m::Int32, b::Int32, input, fill_val::Int32, output)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = b * n * m
    stride = blockDim().x * gridDim().x
    while idx0 < total
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        @inbounds output[idx0 + Int32(1)] = k >= j ? fill_val : input[idx0 + Int32(1)]
        idx0 += stride
    end
    return
end

function lower_diag_mask_kernel!(n::Int32, m::Int32, b::Int32, input, fill_val::Int32, output)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    total = b * n * m
    stride = blockDim().x * gridDim().x
    while idx0 < total
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        @inbounds output[idx0 + Int32(1)] = k <= j ? fill_val : input[idx0 + Int32(1)]
        idx0 += stride
    end
    return
end

function ratio_line(name, h_ref, h_out, fill_val)
    ok = h_ref == h_out
    cnt_fill = count(==(fill_val), h_ref)
    @printf("%s, Mask ratio: %f\n", ok ? "PASS" : "FAIL", cnt_fill / length(h_ref))
    return ok
end

function cpu_sequence!(out, input, seq_lengths, fill_val, m, n, b)
    for idx0 in 0:(b * n * m - 1)
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        out[idx0 + 1] = k >= seq_lengths[j + 1] ? fill_val : input[idx0 + 1]
    end
end

function cpu_window!(out, input, centers, radius, fill_val, m, n, b)
    for idx0 in 0:(b * n * m - 1)
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        out[idx0 + 1] = (k < centers[j + 1] - radius || k > centers[j + 1] + radius) ? fill_val : input[idx0 + 1]
    end
end

function cpu_simple!(out, input, fill_val, m, n, b, pred)
    for idx0 in 0:(b * n * m - 1)
        k = idx0 % m
        j = (idx0 - k) ÷ m % n
        out[idx0 + 1] = pred(k, j) ? fill_val : input[idx0 + 1]
    end
end

function time_kernel!(kernel, args...; repeat)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=GPU_THREADS blocks=args[1] kernel(args[2:end]...)
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1.0e-3 / repeat
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <sequence length> <sequence length> <batch size> <repeat>")
        return 1
    end
    m = parse(Int, args[1])
    n = parse(Int, args[2])
    b = parse(Int, args[3])
    repeat = parse(Int, args[4])
    batch_dim = b <= 0 ? 1 : b
    fill_val = Int32(-1)
    radius = Int32(m ÷ 4)
    data_size = n * m * batch_dim

    println()
    println("M = $m, N = $n, B = $batch_dim")
    rng = MersenneTwister(123)
    seq_lengths = Int32.(rand(rng, 0:(m ÷ 2 - 1), n))
    centers = Int32.(rand(rng, 0:(m - 1), n))
    input = Int32.(rand(rng, 0:(m * n - 1), data_size))
    ref = similar(input)

    d_input = CuArray(input)
    d_output = CuArray{Int32}(undef, data_size)
    d_seq = CuArray(seq_lengths)
    d_centers = CuArray(centers)
    blocks = b <= 0 ? max(1, (n * m) ÷ GPU_THREADS) : n * m
    ok = true

    cpu_sequence!(ref, input, seq_lengths, fill_val, m, n, batch_dim)
    t = time_kernel!(sequence_mask_kernel!, blocks, Int32(n), Int32(m), Int32(batch_dim), d_input, d_seq, fill_val, d_output; repeat)
    @printf("Average execution time of sequenceMask kernel: %f (us)\n", t)
    ok &= ratio_line("sequence", ref, Array(d_output), fill_val)

    cpu_window!(ref, input, centers, radius, fill_val, m, n, batch_dim)
    t = time_kernel!(window_mask_kernel!, blocks, Int32(n), Int32(m), Int32(batch_dim), d_input, d_centers, radius, fill_val, d_output; repeat)
    @printf("Average execution time of windowMask kernel: %f (us)\n", t)
    ok &= ratio_line("window", ref, Array(d_output), fill_val)

    cpu_simple!(ref, input, fill_val, m, n, batch_dim, (k, j) -> k > j)
    t = time_kernel!(upper_mask_kernel!, blocks, Int32(n), Int32(m), Int32(batch_dim), d_input, fill_val, d_output; repeat)
    @printf("Average execution time of upperMask kernel: %f (us)\n", t)
    ok &= ratio_line("upper", ref, Array(d_output), fill_val)

    cpu_simple!(ref, input, fill_val, m, n, batch_dim, (k, j) -> k < j)
    t = time_kernel!(lower_mask_kernel!, blocks, Int32(n), Int32(m), Int32(batch_dim), d_input, fill_val, d_output; repeat)
    @printf("Average execution time of lowerMask kernel: %f (us)\n", t)
    ok &= ratio_line("lower", ref, Array(d_output), fill_val)

    cpu_simple!(ref, input, fill_val, m, n, batch_dim, (k, j) -> k >= j)
    t = time_kernel!(upper_diag_mask_kernel!, blocks, Int32(n), Int32(m), Int32(batch_dim), d_input, fill_val, d_output; repeat)
    @printf("Average execution time of upperDiagMask kernel: %f (us)\n", t)
    ok &= ratio_line("upperDiag", ref, Array(d_output), fill_val)

    cpu_simple!(ref, input, fill_val, m, n, batch_dim, (k, j) -> k <= j)
    t = time_kernel!(lower_diag_mask_kernel!, blocks, Int32(n), Int32(m), Int32(batch_dim), d_input, fill_val, d_output; repeat)
    @printf("Average execution time of lowerDiagMask kernel: %f (us)\n", t)
    ok &= ratio_line("lowerDiag", ref, Array(d_output), fill_val)

    return ok ? 0 : 1
end

exit(main(ARGS))
