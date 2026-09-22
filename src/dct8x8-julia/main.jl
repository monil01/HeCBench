using CUDA
using Printf

const BLOCK_SIZE = 8
const DCT_FORWARD = Int32(666)
const DCT_INVERSE = Int32(777)
const PI32 = Float32(pi)

function usage()
    println("Usage: ", PROGRAM_FILE, " <image width> <image height> <repeat>")
    exit(1)
end

@inline function basis(k, n)
    kk = Float32(k)
    nn = Float32(n)
    return k == 0 ? Float32(0.3535533905932738) :
                    Float32(0.5) * cos(((Float32(2) * nn + Float32(1)) * kk * PI32) / Float32(16))
end

function dct8x8_kernel!(dst, src, stride::Int32, image_h::Int32, image_w::Int32, dir::Int32)
    x0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if x0 < image_w && y0 < image_h
        bx = (x0 ÷ Int32(8)) * Int32(8)
        by = (y0 ÷ Int32(8)) * Int32(8)
        if bx + Int32(7) < image_w && by + Int32(7) < image_h
            u = x0 - bx
            v = y0 - by
            acc = Float32(0)
            if dir == DCT_FORWARD
                n = Int32(0)
                while n < Int32(8)
                    m = Int32(0)
                    while m < Int32(8)
                        sidx = (by + n) * stride + (bx + m) + Int32(1)
                        acc += @inbounds src[sidx] * basis(u, m) * basis(v, n)
                        m += Int32(1)
                    end
                    n += Int32(1)
                end
            else
                n = Int32(0)
                while n < Int32(8)
                    m = Int32(0)
                    while m < Int32(8)
                        sidx = (by + n) * stride + (bx + m) + Int32(1)
                        acc += @inbounds src[sidx] * basis(m, u) * basis(n, v)
                        m += Int32(1)
                    end
                    n += Int32(1)
                end
            end
            didx = y0 * stride + x0 + Int32(1)
            @inbounds dst[didx] = acc
        end
    end
    return
end

function dct8x8_cpu!(dst, src, stride, image_h, image_w, dir)
    fill!(dst, 0.0f0)
    @inbounds for by in 0:BLOCK_SIZE:image_h-BLOCK_SIZE
        for bx in 0:BLOCK_SIZE:image_w-BLOCK_SIZE
            for v in 0:7, u in 0:7
                acc = 0.0f0
                if dir == DCT_FORWARD
                    for n in 0:7, m in 0:7
                        acc += src[(by + n) * stride + (bx + m) + 1] * basis(Int32(u), Int32(m)) * basis(Int32(v), Int32(n))
                    end
                else
                    for n in 0:7, m in 0:7
                        acc += src[(by + n) * stride + (bx + m) + 1] * basis(Int32(m), Int32(u)) * basis(Int32(n), Int32(v))
                    end
                end
                dst[(by + v) * stride + (bx + u) + 1] = acc
            end
        end
    end
    return dst
end

function verify(gpu, cpu, input, stride, image_h, image_w, dir)
    println("Comparing against Host/C++ computation...")
    dct8x8_cpu!(cpu, input, stride, image_h, image_w, dir)
    sum_ref = 0.0
    delta = 0.0
    @inbounds for i in 1:image_h
        base = (i - 1) * stride
        for j in 1:image_w
            c = Float64(cpu[base + j])
            g = Float64(gpu[base + j])
            sum_ref += c * c
            delta += (g - c) * (g - c)
        end
    end
    l2norm = sqrt(delta / max(sum_ref, eps(Float64)))
    @printf("Relative L2 norm: %.3e\n\n", l2norm)
    println(l2norm < 1.0e-6 ? "PASS" : "FAIL")
    return l2norm < 1.0e-6
end

function main()
    length(ARGS) == 3 || usage()
    image_w = parse(Int, ARGS[1])
    image_h = parse(Int, ARGS[2])
    num_iterations = parse(Int, ARGS[3])
    stride = image_w

    println("Allocating and initializing host memory...")
    input = Vector{Float32}(undef, image_h * stride)
    state = UInt32(2009)
    @inbounds for i in eachindex(input)
        state = state * UInt32(1664525) + UInt32(1013904223)
        input[i] = Float32(state >>> 8) / Float32(0x00ffffff)
    end
    output_cpu = zeros(Float32, image_h * stride)
    output_gpu = zeros(Float32, image_h * stride)
    d_input = CuArray(input)
    d_output = CUDA.zeros(Float32, image_h * stride)
    threads = (16, 16)
    blocks = (cld(image_w, threads[1]), cld(image_h, threads[2]))

    println("Performing Forward DCT8x8 of $(image_h) x $(image_w) image on the device\n")
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:num_iterations
        @cuda threads=threads blocks=blocks dct8x8_kernel!(d_output, d_input, Int32(stride), Int32(image_h), Int32(image_w), DCT_FORWARD)
    end
    CUDA.synchronize()
    avg_s = (time_ns() - start) * 1.0e-9 / num_iterations
    @printf("Average DCT8x8 kernel execution time %f (s)\n", avg_s)
    copyto!(output_gpu, Array(d_output))
    ok1 = verify(output_gpu, output_cpu, input, stride, image_h, image_w, DCT_FORWARD)

    println("Performing Inverse DCT8x8 of $(image_h) x $(image_w) image on the device\n")
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:num_iterations
        @cuda threads=threads blocks=blocks dct8x8_kernel!(d_output, d_input, Int32(stride), Int32(image_h), Int32(image_w), DCT_INVERSE)
    end
    CUDA.synchronize()
    avg_s = (time_ns() - start) * 1.0e-9 / num_iterations
    @printf("Average IDCT8x8 kernel execution time %f (s)\n", avg_s)
    copyto!(output_gpu, Array(d_output))
    ok2 = verify(output_gpu, output_cpu, input, stride, image_h, image_w, DCT_INVERSE)
    exit(ok1 && ok2 ? 0 : 1)
end

main()
