using CUDA
using Printf

const RY = Int32(15)
const YG = Int32(6)
const GC_SEG = Int32(4)
const CB = Int32(11)
const BM = Int32(13)
const MR = Int32(6)
const MAXCOLS = Int32(55)

@inline function colorwheel_value(k::Int32, channel::Int32)
    if k < RY
        i = k
        r = Int32(255)
        g = Int32(255) * i ÷ RY
        b = Int32(0)
    elseif k < RY + YG
        i = k - RY
        r = Int32(255) - Int32(255) * i ÷ YG
        g = Int32(255)
        b = Int32(0)
    elseif k < RY + YG + GC_SEG
        i = k - RY - YG
        r = Int32(0)
        g = Int32(255)
        b = Int32(255) * i ÷ GC_SEG
    elseif k < RY + YG + GC_SEG + CB
        i = k - RY - YG - GC_SEG
        r = Int32(0)
        g = Int32(255) - Int32(255) * i ÷ CB
        b = Int32(255)
    elseif k < RY + YG + GC_SEG + CB + BM
        i = k - RY - YG - GC_SEG - CB
        r = Int32(255) * i ÷ BM
        g = Int32(0)
        b = Int32(255)
    else
        i = k - RY - YG - GC_SEG - CB - BM
        r = Int32(255)
        g = Int32(0)
        b = Int32(255) - Int32(255) * i ÷ MR
    end

    return channel == Int32(0) ? r : (channel == Int32(1) ? g : b)
end

@inline function compute_color_channel(fx::Float32, fy::Float32, out_channel::Int32)
    rad = sqrt(fx * fx + fy * fy)
    a = atan(-fy, -fx) / Float32(pi)
    fk = (a + 1.0f0) * 0.5f0 * Float32(MAXCOLS - Int32(1))
    k0 = Int32(trunc(fk))
    k1 = k0 + Int32(1)
    if k1 == MAXCOLS
        k1 = Int32(0)
    end
    f = fk - Float32(k0)

    cw_channel = Int32(2) - out_channel
    col0 = Float32(colorwheel_value(k0, cw_channel)) / 255.0f0
    col1 = Float32(colorwheel_value(k1, cw_channel)) / 255.0f0
    col = (1.0f0 - f) * col0 + f * col1
    if rad <= 1.0f0
        col = 1.0f0 - rad * (1.0f0 - col)
    else
        col *= 0.75f0
    end
    return UInt8(trunc(Int32, 255.0f0 * col))
end

function color_kernel!(pix, size::Int32, half_size::Int32, range::Float32, truerange::Float32)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)

    if y < size && x < size
        fx = Float32(x) / Float32(half_size) * range - range
        fy = Float32(y) / Float32(half_size) * range - range
        if x == half_size || y == half_size
            return
        end
        idx = (y * size + x) * Int32(3)
        @inbounds pix[idx + Int32(1)] = compute_color_channel(fx / truerange, fy / truerange, Int32(0))
        @inbounds pix[idx + Int32(2)] = compute_color_channel(fx / truerange, fy / truerange, Int32(1))
        @inbounds pix[idx + Int32(3)] = compute_color_channel(fx / truerange, fy / truerange, Int32(2))
    end
    return
end

function compute_color_host!(pix::Vector{UInt8}, idx::Int, fx::Float32, fy::Float32)
    pix[idx + 1] = compute_color_channel(fx, fy, Int32(0))
    pix[idx + 2] = compute_color_channel(fx, fy, Int32(1))
    pix[idx + 3] = compute_color_channel(fx, fy, Int32(2))
end

function run_colorwheel(truerange_arg::Float32, size::Int, repeat::Int)
    range = 1.04f0 * truerange_arg
    half_size = size ÷ 2
    img_size = size * size * 3

    pix = zeros(UInt8, img_size)
    res = Vector{UInt8}(undef, img_size)

    for y in 0:size-1
        for x in 0:size-1
            fx = Float32(x) / Float32(half_size) * range - range
            fy = Float32(y) / Float32(half_size) * range - range
            if x == half_size || y == half_size
                continue
            end
            idx = (y * size + x) * 3
            compute_color_host!(pix, idx, fx / truerange_arg, fy / truerange_arg)
        end
    end

    println("Start execution on a device")
    d_pix = CUDA.zeros(UInt8, img_size)
    threads = (16, 16)
    blocks = (cld(size, 16), cld(size, 16))

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks color_kernel!(
            d_pix, Int32(size), Int32(half_size), range, truerange_arg)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) * 1e-6 / repeat
    @printf("Average kernel execution time : %f (ms)\n", elapsed_ms)

    copyto!(res, d_pix)
    tolerance = 1
    fail = false
    max_error = 0
    for i in eachindex(pix)
        e = abs(Int(res[i]) - Int(pix[i]))
        if e > tolerance
            fail = true
            if e > max_error
                max_error = e
            end
        end
    end

    if fail
        @printf("Verification failed. Maximum error between host and device results: %d\n", max_error)
    else
        println("PASS")
    end
    return fail ? 1 : 0
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <range> <size> <repeat>")
        return 1
    end
    truerange = parse(Float32, ARGS[1])
    size = parse(Int, ARGS[2])
    repeat = parse(Int, ARGS[3])
    return run_colorwheel(truerange, size, repeat)
end

exit(main())
