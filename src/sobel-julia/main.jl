using CUDA
using Printf

# Julia port of sobel-cuda benchmark.
# Simplification: instead of parsing a .bmp we synthesize a deterministic
# uchar4 image of the same 512x512 size and compare device output against
# a Julia CPU reference. Kernel math is identical to the CUDA original.

const WIDTH  = 512
const HEIGHT = 512

@inline function clamp_u8(v::Float32)
    if v > 255.0f0
        return UInt8(255)
    elseif v < 0.0f0
        return UInt8(0)
    else
        return UInt8(unsafe_trunc(Int32, v))
    end
end

# Input layout: 4 planes x[c], y[c], z[c], w[c] -> we use a single 4*N UInt8 array
# indexed as inp[4*(c-1)+ch]
function sobel_kernel!(inp, out, width::Int32, height::Int32)
    x = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x - 1)  # 0-based
    y = Int32((blockIdx().y - 1)) * Int32(blockDim().y) + Int32(threadIdx().y - 1)
    if x >= Int32(1) && x < width - Int32(1) && y >= Int32(1) && y < height - Int32(1)
        c = x + y * width  # 0-based pixel index
        @inbounds begin
            # helper macros not available; inline
            function px(off, ch)
                idx = Int32(4) * (c + off) + Int32(ch)  # 0-based byte offset
                return Float32(inp[idx + Int32(1)])
            end
            for ch in Int32(0):Int32(3)
                i00 = px(-Int32(1) - width, ch)
                i01 = px(-width, ch)
                i02 = px(Int32(1) - width, ch)
                i10 = px(-Int32(1), ch)
                i12 = px(Int32(1), ch)
                i20 = px(-Int32(1) + width, ch)
                i21 = px(width, ch)
                i22 = px(Int32(1) + width, ch)
                Gx = i00 + 2.0f0 * i10 + i20 - i02 - 2.0f0 * i12 - i22
                Gy = i00 - i20 + 2.0f0 * i01 - 2.0f0 * i21 + i02 - i22
                mag = sqrt(Gx * Gx + Gy * Gy) / 2.0f0
                out_idx = Int32(4) * c + ch + Int32(1)
                out[out_idx] = clamp_u8(mag)
            end
        end
    end
    return
end

function sobel_cpu!(inp::Vector{UInt8}, out::Vector{UInt8}, width, height)
    for y in 1:(height-2)
        for x in 1:(width-2)  # 0-based: x=1..width-2, y=1..height-2
            c = x + y * width  # 0-based pixel idx
            for ch in 0:3
                function px(off, ch)
                    idx = 4 * (c + off) + ch  # 0-based
                    return Float32(inp[idx + 1])
                end
                i00 = px(-1 - width, ch); i01 = px(-width, ch); i02 = px(1 - width, ch)
                i10 = px(-1, ch);                             i12 = px(1, ch)
                i20 = px(-1 + width, ch); i21 = px(width, ch); i22 = px(1 + width, ch)
                Gx = i00 + 2f0 * i10 + i20 - i02 - 2f0 * i12 - i22
                Gy = i00 - i20 + 2f0 * i01 - 2f0 * i21 + i02 - i22
                mag = sqrt(Gx*Gx + Gy*Gy) / 2f0
                out_idx = 4 * c + ch + 1
                out[out_idx] = clamp_u8(mag)
            end
        end
    end
end

function main()
    if length(ARGS) < 1
        println("Usage: main.jl [<bmp path ignored>] <repeat>")
        return 1
    end
    # If two args given, ignore first (path), use second (repeat)
    repeat_n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : parse(Int, ARGS[1])

    width, height = WIDTH, HEIGHT
    N = width * height
    total = N * 4

    # Deterministic synthetic uchar4 image (LCG)
    state = UInt64(0x1234567)
    inp = Vector{UInt8}(undef, total)
    for i in 1:total
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        inp[i] = UInt8((state >> 32) & UInt64(0xFF))
    end

    @printf("Image height = %d and width = %d\n", height, width)

    d_inp = CuArray(inp)
    d_out = CUDA.zeros(UInt8, total)

    threads = (16, 16)
    blocks  = (cld(width, 16), cld(height, 16))

    @cuda threads=threads blocks=blocks sobel_kernel!(d_inp, d_out, Int32(width), Int32(height))
    CUDA.synchronize()

    @printf("Executing kernel for %d iterations", repeat_n)
    print("-------------------------------------------\n")
    t0 = time_ns()
    for _ in 1:repeat_n
        @cuda threads=threads blocks=blocks sobel_kernel!(d_inp, d_out, Int32(width), Int32(height))
    end
    CUDA.synchronize()
    ktime_us = (time_ns() - t0) * 1e-3 / repeat_n
    @printf("Average kernel execution time: %f (us)\n", ktime_us)

    out_dev = Array(d_out)

    out_ref = zeros(UInt8, total)
    sobel_cpu!(inp, out_ref, width, height)

    err = 0.0f0
    ref_norm = 0.0f0
    for i in 2:total
        d = Float32(out_ref[i]) - Float32(out_dev[i])
        err += d * d
        ref_norm += Float32(out_ref[i]) * Float32(out_ref[i])
    end
    rel = sqrt(err) / sqrt(ref_norm)
    println(rel < 1f-6 ? "PASS" : "FAIL")
    return 0
end

main()
