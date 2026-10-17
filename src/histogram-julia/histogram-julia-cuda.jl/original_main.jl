using CUDA
using Printf

# Julia port of histogram-cuda (simplified, follows the Triton port).
# Deterministic random 1920x1080 uchar4 image; per-channel 256-bin histogram.

const NUM_BINS = 256
const ACTIVE_CHANNELS = 3
const NUM_CHANNELS = 4

function histogram_kernel!(pixels, hist, num_pixels::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)  # 1-based pixel idx
    @inbounds if i <= num_pixels
        base = Int32(4) * (i - Int32(1))  # 0-based byte offset
        r = Int32(pixels[base + Int32(1)])
        g = Int32(pixels[base + Int32(2)])
        b = Int32(pixels[base + Int32(3)])
        # Julia atomic_add on CuArray uses CUDA.@atomic
        CUDA.@atomic hist[0 * Int32(256) + r + Int32(1)] += Int32(1)
        CUDA.@atomic hist[1 * Int32(256) + g + Int32(1)] += Int32(1)
        CUDA.@atomic hist[2 * Int32(256) + b + Int32(1)] += Int32(1)
    end
    return
end

function parse_args()
    iters = 100
    width = 1920
    height = 1080
    entropy = 0
    for a in ARGS
        if startswith(a, "--i=")
            iters = parse(Int, a[5:end])
        elseif startswith(a, "--width=")
            width = parse(Int, a[9:end])
        elseif startswith(a, "--height=")
            height = parse(Int, a[10:end])
        elseif startswith(a, "--entropy=")
            entropy = parse(Int, a[11:end])
        end
    end
    return iters, width, height, entropy
end

function main()
    iters, width, height, entropy = parse_args()
    @printf("Random image: entropy-reduction(%d) width(%d) height(%d)\n", entropy, width, height)

    # Deterministic LCG uchar4 image
    num_pixels = width * height
    total = num_pixels * NUM_CHANNELS
    img = Vector{UInt8}(undef, total)
    state = UInt64(0xABCDEF)
    for i in 1:total
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        img[i] = UInt8((state >> 32) & UInt64(0xFF))
    end

    d_pixels = CuArray(img)
    d_hist = CUDA.zeros(Int32, ACTIVE_CHANNELS * NUM_BINS)

    # CPU reference
    ref = zeros(Int64, ACTIVE_CHANNELS, NUM_BINS)
    for p in 1:num_pixels
        base = 4 * (p - 1)
        for c in 1:ACTIVE_CHANNELS
            v = Int(img[base + c])  # 0..255
            ref[c, v + 1] += 1
        end
    end

    threads = 512
    blocks  = cld(num_pixels, threads)

    # Warmup
    CUDA.fill!(d_hist, Int32(0))
    @cuda threads=threads blocks=blocks histogram_kernel!(d_pixels, d_hist, Int32(num_pixels))
    CUDA.synchronize()

    t0 = time_ns()
    for _ in 1:iters
        CUDA.fill!(d_hist, Int32(0))
        @cuda threads=threads blocks=blocks histogram_kernel!(d_pixels, d_hist, Int32(num_pixels))
    end
    CUDA.synchronize()
    ktime_ms = (time_ns() - t0) * 1e-6 / iters
    @printf("Average kernel execution time: %f (ms)\n", ktime_ms)

    got = Array(d_hist)
    ok = true
    for c in 1:ACTIVE_CHANNELS, b in 1:NUM_BINS
        if got[(c-1)*NUM_BINS + b] != ref[c, b]
            ok = false
        end
    end
    println(ok ? "PASS" : "FAIL")
    return 0
end

main()
