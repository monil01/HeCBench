using CUDA
using Printf

const TPB = 256
const TILE_M = 16
const TILE_N = 256

function add_kernel!(a, b, c, total::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= total
        @inbounds c[i] = a[i] + b[i]
        i += stride
    end
    return
end

function tv_add_kernel!(a, b, c, m::Int32, n::Int32)
    row0 = (blockIdx().y - Int32(1)) * Int32(TILE_M)
    col0 = (blockIdx().x - Int32(1)) * Int32(TILE_N)
    tid0 = threadIdx().x - Int32(1)
    total = Int32(TILE_M * TILE_N)
    stride = blockDim().x
    k = tid0
    while k < total
        r = k ÷ Int32(TILE_N)
        col = k - r * Int32(TILE_N)
        row = row0 + r
        cc = col0 + col
        if row < m && cc < n
            idx = row * n + cc + Int32(1)
            @inbounds c[idx] = a[idx] + b[idx]
        end
        k += stride
    end
    return
end

function fill_values(n::Int)
    ccall(:srand, Cvoid, (Cuint,), UInt32(42))
    ref = Vector{Float32}(undef, n)
    a = Vector{Float16}(undef, n)
    b = Vector{Float16}(undef, n)
    for i in 1:n
        ta = (ccall(:rand, Cint, ()) % 2) != 0 ? 1.0f0 : -1.0f0
        tb = (ccall(:rand, Cint, ()) % 2) != 0 ? 1.0f0 : -1.0f0
        ref[i] = ta + tb
        a[i] = Float16(ta)
        b[i] = Float16(tb)
    end
    return ref, a, b
end

function allclose(ref, got)
    for i in eachindex(ref)
        r = ref[i]
        g = Float32(got[i])
        if abs(r - g) > 1.0f-2 + 1.0f-5 * abs(r)
            return false
        end
    end
    return true
end

function print_result(label, pass, avg_ms, gbps, repeat)
    @printf("  correctness : %s\n", pass ? "PASS" : "FAIL")
    @printf("  avg  time   : %.3f ms  (%d iters)\n", avg_ms, repeat)
    @printf("  bandwidth   : %.1f GB/s\n\n", gbps)
end

function time_add!(kernel, d_a, d_b, d_c, args...; repeat, blocks, threads, bytes_moved)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks kernel(d_a, d_b, d_c, args...)
    end
    CUDA.synchronize()
    avg_ms = (time_ns() - t0) * 1.0e-6 / repeat
    gbps = bytes_moved / (avg_ms * 1.0e-3) / 1.0e9
    return avg_ms, gbps
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <number of rows> <number of columns> <repeat>")
        return 1
    end
    m = parse(Int, args[1])
    n = parse(Int, args[2])
    repeat = parse(Int, args[3])
    total = m * n
    ref, h_a, h_b = fill_values(total)
    d_a = CuArray(h_a)
    d_b = CuArray(h_b)
    d_c = CUDA.zeros(Float16, total)
    bytes_moved = 3.0 * total * sizeof(Float16)

    println("=== Base (one thread per element) ===")
    blocks = cld(total, TPB)
    @cuda threads=TPB blocks=blocks add_kernel!(d_a, d_b, d_c, Int32(total))
    pass1 = allclose(ref, Array(d_c))
    r1_ms, r1_bw = time_add!(add_kernel!, d_a, d_b, d_c, Int32(total);
                             repeat=repeat, blocks=blocks, threads=TPB, bytes_moved=bytes_moved)
    print_result("base", pass1, r1_ms, r1_bw, repeat)

    println("=== Vectorised (8x fp16 = 128-bit load per thread) ===")
    @cuda threads=TPB blocks=blocks add_kernel!(d_a, d_b, d_c, Int32(total))
    pass2 = allclose(ref, Array(d_c))
    r2_ms, r2_bw = time_add!(add_kernel!, d_a, d_b, d_c, Int32(total);
                             repeat=repeat, blocks=blocks, threads=TPB, bytes_moved=bytes_moved)
    print_result("vectorized", pass2, r2_ms, r2_bw, repeat)

    println("=== TV-layout (block tile 16x256, 8x fp16 per thread per row) ===")
    tv_blocks = (cld(n, TILE_N), cld(m, TILE_M))
    @cuda threads=128 blocks=tv_blocks tv_add_kernel!(d_a, d_b, d_c, Int32(m), Int32(n))
    pass3 = allclose(ref, Array(d_c))
    r3_ms, r3_bw = time_add!(tv_add_kernel!, d_a, d_b, d_c, Int32(m), Int32(n);
                             repeat=repeat, blocks=tv_blocks, threads=128, bytes_moved=bytes_moved)
    print_result("tv_layout", pass3, r3_ms, r3_bw, repeat)

    println("-----------------------------------------------------")
    @printf("%-12s  %9s  %10s\n", "Kernel", "Avg(ms)", "BW(GB/s)")
    println("-----------------------------------------------------")
    @printf("%-12s  %9.3f  %10.1f\n", "base", r1_ms, r1_bw)
    @printf("%-12s  %9.3f  %10.1f\n", "vectorized", r2_ms, r2_bw)
    @printf("%-12s  %9.3f  %10.1f\n", "tv_layout", r3_ms, r3_bw)
    println("-----------------------------------------------------")
    return (pass1 && pass2 && pass3) ? 0 : 1
end

exit(main(ARGS))
