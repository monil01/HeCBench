using CUDA
using Printf
using Random

const BLOCK_SIZES = (64, 128, 256, 512, 1024)

function silu_forward_kernel!(x, out, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= n
        @inbounds xv = x[i]
        @inbounds out[i] = xv / (1.0f0 + exp(-xv))
        i += stride
    end
    return
end

function silu_backward_kernel!(dout, x, dx, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= n
        @inbounds xv = x[i]
        @inbounds ov = dout[i]
        expx = exp(-xv)
        grad = (1.0f0 + xv * expx / (1.0f0 + expx)) / (1.0f0 + expx)
        @inbounds dx[i] = ov * grad
        i += stride
    end
    return
end

function silu_backward2_kernel!(dout, x, dx, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= n
        @inbounds xv = x[i]
        sig = 1.0f0 / (1.0f0 + exp(-xv))
        grad = sig * (1.0f0 + xv * (1.0f0 - sig))
        @inbounds dx[i] = dout[i] * grad
        i += stride
    end
    return
end

function forward_ref!(x, out)
    @inbounds for i in eachindex(x)
        xv = x[i]
        out[i] = xv / (1.0f0 + exp(-xv))
    end
end

function backward_ref!(dout, x, dx)
    @inbounds for i in eachindex(x)
        xv = x[i]
        expx = exp(-xv)
        grad = (1.0f0 + xv * expx / (1.0f0 + expx)) / (1.0f0 + expx)
        dx[i] = dout[i] * grad
    end
end

function validate_result(device_result, cpu_reference, name::String)
    got = Array(device_result)
    nfaults = 0
    for i in eachindex(got)
        r = cpu_reference[i]
        g = got[i]
        if abs(r - g) > 1.0f-4 && isfinite(r)
            @printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i - 1, r, g)
            nfaults += 1
            if nfaults >= 10
                break
            end
        end
    end
    println(nfaults == 0 ? "PASS" : "FAIL")
    return nfaults == 0
end

function run_kernel!(kernel, args...; n::Int, block_size::Int, repeat::Int)
    blocks = cld(n, block_size)
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=block_size blocks=blocks kernel(args..., Int32(n))
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-6 / repeat
end

function main(args)
    if length(args) != 5
        println("Usage: main.jl <batch size> <number of channels> <height> <width> <repeat>")
        return 1
    end
    b = parse(Int, args[1])
    c = parse(Int, args[2])
    h = parse(Int, args[3])
    w = parse(Int, args[4])
    repeat = parse(Int, args[5])
    n = b * c * h * w

    rng = MersenneTwister(0)
    x = rand(rng, Float32, n) .* 2.0f0 .- 1.0f0
    dout = rand(rng, Float32, n) .* 2.0f0 .- 1.0f0
    out_ref = Vector{Float32}(undef, n)
    dx_ref = Vector{Float32}(undef, n)
    forward_ref!(x, out_ref)
    backward_ref!(dout, x, dx_ref)

    d_x = CuArray(x)
    d_dout = CuArray(dout)
    d_out = CUDA.zeros(Float32, n)
    d_dx = CUDA.zeros(Float32, n)

    println("Checking forward pass")
    for bs in BLOCK_SIZES
        @printf("Checking block size %d\n", bs)
        @cuda threads=bs blocks=cld(n, bs) silu_forward_kernel!(d_x, d_out, Int32(n))
        validate_result(d_out, out_ref, "out")
    end

    println("Checking forward2 pass")
    for bs in BLOCK_SIZES
        @printf("Checking block size %d\n", bs)
        @cuda threads=bs blocks=cld(n, bs) silu_forward_kernel!(d_x, d_out, Int32(n))
        validate_result(d_out, out_ref, "out")
    end

    println("Checking backward pass")
    for bs in BLOCK_SIZES
        @printf("Checking block size %d\n", bs)
        @cuda threads=bs blocks=cld(n, bs) silu_backward_kernel!(d_dout, d_x, d_dx, Int32(n))
        validate_result(d_dx, dx_ref, "dx")
    end

    println("Checking backward2 pass")
    for bs in BLOCK_SIZES
        @printf("Checking block size %d\n", bs)
        @cuda threads=bs blocks=cld(n, bs) silu_backward2_kernel!(d_dout, d_x, d_dx, Int32(n))
        validate_result(d_dx, dx_ref, "dx")
    end

    println("Checking backward3 pass")
    for bs in BLOCK_SIZES
        @printf("Checking block size %d\n", bs)
        @cuda threads=bs blocks=cld(n, bs) silu_backward2_kernel!(d_dout, d_x, d_dx, Int32(n))
        validate_result(d_dx, dx_ref, "dx")
    end

    println()
    println("Forward pass benchmarks:")
    for bs in BLOCK_SIZES
        elapsed = run_kernel!(silu_forward_kernel!, d_x, d_out; n=n, block_size=bs, repeat=repeat)
        @printf("block_size %4d | time %.4f ms\n", bs, elapsed)
    end
    println()
    println("Forward2 pass benchmarks:")
    for bs in BLOCK_SIZES
        elapsed = run_kernel!(silu_forward_kernel!, d_x, d_out; n=n, block_size=bs, repeat=repeat)
        @printf("block_size %4d | time %.4f ms\n", bs, elapsed)
    end
    println()
    println("Backward pass benchmarks:")
    for bs in BLOCK_SIZES
        elapsed = run_kernel!(silu_backward_kernel!, d_dout, d_x, d_dx; n=n, block_size=bs, repeat=repeat)
        @printf("block_size %4d | time %.4f ms\n", bs, elapsed)
    end
    println()
    println("Backward2 pass benchmarks:")
    for bs in BLOCK_SIZES
        elapsed = run_kernel!(silu_backward2_kernel!, d_dout, d_x, d_dx; n=n, block_size=bs, repeat=repeat)
        @printf("block_size %4d | time %.4f ms\n", bs, elapsed)
    end
    println()
    println("Backward3 pass benchmarks:")
    for bs in BLOCK_SIZES
        elapsed = run_kernel!(silu_backward2_kernel!, d_dout, d_x, d_dx; n=n, block_size=bs, repeat=repeat)
        @printf("block_size %4d | time %.4f ms\n", bs, elapsed)
    end
    return 0
end

exit(main(ARGS))
