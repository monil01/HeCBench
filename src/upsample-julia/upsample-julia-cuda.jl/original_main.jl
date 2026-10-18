using CUDA
using Printf
using Random

ceil_div(a::Integer, b::Integer) = (a + b - 1) ÷ b

function upsample_forward_kernel!(x, out, B::UInt64, C::UInt64, H::UInt64, W::UInt64, total::UInt64)
    flat = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if flat >= total
        return
    end
    img_size = H * W
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    img_out_size = h_out * w_out
    b = flat ÷ (C * img_size)
    c = (flat ÷ img_size) % C
    i = (flat ÷ W) % H
    j = flat % W
    x_idx = b * C * img_size + c * img_size + i * W + j
    out_base = b * C * img_out_size + c * img_out_size + UInt64(2) * i * w_out + UInt64(2) * j
    @inbounds val = x[Int(x_idx) + 1]
    @inbounds out[Int(out_base) + 1] = val
    @inbounds out[Int(out_base) + 2] = val
    @inbounds out[Int(out_base + w_out) + 1] = val
    @inbounds out[Int(out_base + w_out) + 2] = val
    return
end

function upsample_forward_kernel2!(x, out, B::UInt64, C::UInt64, H::UInt64, W::UInt64)
    in_x = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    in_y = UInt64((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    bc = UInt64(blockIdx().z - Int32(1))
    if in_x >= W || in_y >= H
        return
    end
    b = bc ÷ C
    c = bc % C
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    @inbounds val = x[Int((b * C + c) * H * W + in_y * W + in_x) + 1]
    out_base = (b * C + c) * h_out * w_out + (in_y * UInt64(2)) * w_out + in_x * UInt64(2)
    @inbounds out[Int(out_base) + 1] = val
    @inbounds out[Int(out_base) + 2] = val
    @inbounds out[Int(out_base + w_out) + 1] = val
    @inbounds out[Int(out_base + w_out) + 2] = val
    return
end

function upsample_backward_kernel!(dout, dx, B::UInt64, C::UInt64, H::UInt64, W::UInt64, total::UInt64)
    flat = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    if flat >= total
        return
    end
    img_size = H * W
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    img_out_size = h_out * w_out
    b = flat ÷ (C * img_size)
    c = (flat ÷ img_size) % C
    i = (flat ÷ W) % H
    j = flat % W
    out_base = b * C * img_out_size + c * img_out_size + UInt64(2) * i * w_out + UInt64(2) * j
    @inbounds dx[Int(flat) + 1] = dout[Int(out_base) + 1] + dout[Int(out_base) + 2] +
                                  dout[Int(out_base + w_out) + 1] + dout[Int(out_base + w_out) + 2]
    return
end

function upsample_backward_kernel2!(dout, dx, B::UInt64, C::UInt64, H::UInt64, W::UInt64)
    in_x = UInt64((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1))
    in_y = UInt64((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1))
    bc = UInt64(blockIdx().z - Int32(1))
    if in_x >= W || in_y >= H
        return
    end
    b = bc ÷ C
    c = bc % C
    h_out = H * UInt64(2)
    w_out = W * UInt64(2)
    out_base = b * C * h_out * w_out + c * h_out * w_out + UInt64(2) * in_y * w_out + UInt64(2) * in_x
    @inbounds dx[Int((b * C + c) * H * W + in_y * W + in_x) + 1] =
        dout[Int(out_base) + 1] + dout[Int(out_base) + 2] +
        dout[Int(out_base + w_out) + 1] + dout[Int(out_base + w_out) + 2]
    return
end

function forward_reference(x, B, C, H, W)
    out = Vector{Float32}(undef, B * C * H * W * 4)
    H_out = H * 2
    W_out = W * 2
    img_size = H * W
    img_out_size = H_out * W_out
    for b in 0:(B - 1), c in 0:(C - 1), i in 0:(H - 1), j in 0:(W - 1)
        val = x[b * C * img_size + c * img_size + i * W + j + 1]
        offset = b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j
        out[offset + 1] = val
        out[offset + 2] = val
        out[offset + W_out + 1] = val
        out[offset + W_out + 2] = val
    end
    return out
end

function backward_reference(dout, B, C, H, W)
    dx = Vector{Float32}(undef, B * C * H * W)
    H_out = H * 2
    W_out = W * 2
    img_size = H * W
    img_out_size = H_out * W_out
    for b in 0:(B - 1), c in 0:(C - 1), i in 0:(H - 1), j in 0:(W - 1)
        dx_offset = b * C * img_size + c * img_size + i * W + j
        out_offset = b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j
        dx[dx_offset + 1] = dout[out_offset + 1] + dout[out_offset + 2] +
                            dout[out_offset + W_out + 1] + dout[out_offset + W_out + 2]
    end
    return dx
end

function validate_result(device_result, reference, n)
    out = Array(device_result)
    nfaults = 0
    for i in 1:n
        if abs(reference[i] - out[i]) > Float32(1.0f-4) && isfinite(reference[i])
            nfaults += 1
            nfaults >= 10 && break
        end
    end
    println(nfaults == 0 ? "PASS" : "FAIL")
end

function benchmark(repeat, f)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        f()
    end
    CUDA.synchronize()
    return (time_ns() - start) * 1e-6 / repeat
end

function main(args)
    if length(args) != 5
        println("Usage: main.jl <batch size> <number of channels> <height> <width> <repeat>")
        return 1
    end
    B, C, H, W, repeat = parse.(Int, args)
    S = B * C * H * W
    rng = MersenneTwister(0)
    x = rand(rng, Float32, S) .* Float32(2) .- Float32(1)
    dout = rand(rng, Float32, S * 4) .* Float32(2) .- Float32(1)
    out_ref = forward_reference(x, B, C, H, W)
    dx_ref = backward_reference(dout, B, C, H, W)

    d_x = CuArray(x)
    d_out = CUDA.zeros(Float32, S * 4)
    d_dout = CuArray(dout)
    d_dx = CUDA.zeros(Float32, S)

    Bu, Cu, Hu, Wu = UInt64(B), UInt64(C), UInt64(H), UInt64(W)
    Su = UInt64(S)

    block_sizes = [32, 64, 128, 256, 512, 1024]
    println("Checking forward pass")
    for block_size in block_sizes
        println("Checking block size $block_size")
        @cuda threads=block_size blocks=ceil_div(S, block_size) upsample_forward_kernel!(d_x, d_out, Bu, Cu, Hu, Wu, Su)
        CUDA.synchronize()
        validate_result(d_out, out_ref, B * C * (H ÷ 2) * (W ÷ 2))
    end

    println("Forward1 pass benchmarks:")
    for block_size in block_sizes
        elapsed = benchmark(repeat, () -> @cuda(threads=block_size, blocks=ceil_div(S, block_size),
                                                upsample_forward_kernel!(d_x, d_out, Bu, Cu, Hu, Wu, Su)))
        gflops = Float32(S) / elapsed * 1.0f3 / 1.0f9
        @printf("block_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops)
    end

    println("\n─────────────────────────────────────────────────────")
    block2d_sizes = [8, 16, 32]
    for block_size in block2d_sizes
        println("Checking block size $block_size")
        @cuda threads=(block_size, block_size, 1) blocks=(ceil_div(W, block_size), ceil_div(H, block_size), B * C) upsample_forward_kernel2!(d_x, d_out, Bu, Cu, Hu, Wu)
        CUDA.synchronize()
        validate_result(d_out, out_ref, B * C * (H ÷ 2) * (W ÷ 2))
    end

    println("Forward2 pass benchmarks:")
    for block_size in block2d_sizes
        elapsed = benchmark(repeat, () -> @cuda(threads=(block_size, block_size, 1), blocks=(ceil_div(W, block_size), ceil_div(H, block_size), B * C),
                                                upsample_forward_kernel2!(d_x, d_out, Bu, Cu, Hu, Wu)))
        gflops = Float32(S) / elapsed * 1.0f3 / 1.0f9
        @printf("block2D_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops)
    end

    println("\n─────────────────────────────────────────────────────")
    println("Checking backward pass")
    for block_size in block_sizes
        println("Checking block size $block_size")
        @cuda threads=block_size blocks=ceil_div(S, block_size) upsample_backward_kernel!(d_dout, d_dx, Bu, Cu, Hu, Wu, Su)
        CUDA.synchronize()
        validate_result(d_dx, dx_ref, S)
    end

    println("\nBackward pass benchmarks:")
    for block_size in block_sizes
        elapsed = benchmark(repeat, () -> @cuda(threads=block_size, blocks=ceil_div(S, block_size),
                                                upsample_backward_kernel!(d_dout, d_dx, Bu, Cu, Hu, Wu, Su)))
        gflops = Float32(S) / elapsed * 1.0f3 / 1.0f9
        @printf("block_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops)
    end

    println("\n─────────────────────────────────────────────────────")
    println("Checking backward2 pass")
    for block_size in block2d_sizes
        println("Checking block size $block_size")
        @cuda threads=(block_size, block_size, 1) blocks=(ceil_div(W, block_size), ceil_div(H, block_size), B * C) upsample_backward_kernel2!(d_dout, d_dx, Bu, Cu, Hu, Wu)
        CUDA.synchronize()
        validate_result(d_dx, dx_ref, S)
    end

    println("\nBackward2 pass benchmarks:")
    for block_size in block2d_sizes
        elapsed = benchmark(repeat, () -> @cuda(threads=(block_size, block_size, 1), blocks=(ceil_div(W, block_size), ceil_div(H, block_size), B * C),
                                                upsample_backward_kernel2!(d_dout, d_dx, Bu, Cu, Hu, Wu)))
        gflops = Float32(S) / elapsed * 1.0f3 / 1.0f9
        @printf("block2D_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed, gflops)
    end
    return 0
end

exit(main(ARGS))
