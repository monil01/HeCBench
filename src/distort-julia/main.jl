using CUDA
using Printf

struct Properties
    K::Float32
    centerX::Float32
    centerY::Float32
    width::Int32
    height::Int32
    thresh::Float32
    xshift::Float32
    yshift::Float32
    xscale::Float32
    yscale::Float32
end

function libc_srand(seed::Integer)
    ccall(:srand, Cvoid, (Cuint,), Cuint(seed))
end

function libc_rand()
    ccall(:rand, Cint, ())
end

function calc_shift(x1::Float32, x2::Float32, cx::Float32, k::Float32, thresh::Float32)
    while true
        x3 = x1 + (x2 - x1) * Float32(0.5)
        result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)))
        result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)))
        if result1 > -thresh && result1 < thresh
            return x1
        elseif result3 < 0
            x1 = x3
        else
            x2 = x3
        end
    end
end

function make_properties(width::Int, height::Int, k::Float32)
    center_x = Float32(width ÷ 2)
    center_y = Float32(height ÷ 2)
    thresh = Float32(1)
    xshift = calc_shift(0f0, center_x - 1f0, center_x, k, thresh)
    xshift_2 = calc_shift(0f0, Float32(width) - center_x - 1f0, Float32(width) - center_x, k, thresh)
    yshift = calc_shift(0f0, center_y - 1f0, center_y, k, thresh)
    yshift_2 = calc_shift(0f0, Float32(height) - center_y - 1f0, Float32(height) - center_y, k, thresh)
    xscale = (Float32(width) - xshift - xshift_2) / Float32(width)
    yscale = (Float32(height) - yshift - yshift_2) / Float32(height)
    return Properties(k, center_x, center_y, Int32(width), Int32(height), thresh,
                      xshift, yshift, xscale, yscale)
end

@inline function radial_x(x::Float32, y::Float32, prop::Properties)
    sx = x * prop.xscale + prop.xshift
    sy = y * prop.yscale + prop.yshift
    dx = sx - prop.centerX
    dy = sy - prop.centerY
    return sx + dx * prop.K * (dx * dx + dy * dy)
end

@inline function radial_y(x::Float32, y::Float32, prop::Properties)
    sx = x * prop.xscale + prop.xshift
    sy = y * prop.yscale + prop.yshift
    dx = sx - prop.centerX
    dy = sy - prop.centerY
    return sy + dy * prop.K * (dx * dx + dy * dy)
end

@inline function interp_channel(src, c::Int32, idx0::Float32, idx1::Float32, prop::Properties)
    if idx0 < 0f0 || idx1 < 0f0 || idx0 > Float32(prop.height - Int32(1)) || idx1 > Float32(prop.width - Int32(1))
        return UInt8(0)
    end
    h0 = Int32(floor(idx0))
    h1 = Int32(ceil(idx0))
    w0 = Int32(floor(idx1))
    w1 = Int32(ceil(idx1))
    x = idx0 - Float32(h0)
    y = idx1 - Float32(w0)
    base1 = h0 * prop.width + w0 + Int32(1)
    base2 = h0 * prop.width + w1 + Int32(1)
    base3 = h1 * prop.width + w1 + Int32(1)
    base4 = h1 * prop.width + w0 + Int32(1)
    v = Float32(@inbounds src[c, base1]) * (1f0 - x) * (1f0 - y) +
        Float32(@inbounds src[c, base2]) * (1f0 - x) * y +
        Float32(@inbounds src[c, base3]) * x * y +
        Float32(@inbounds src[c, base4]) * x * (1f0 - y)
    return UInt8(trunc(Int32, v))
end

function distort_kernel!(src, dst, prop::Properties)
    w = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    h = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if w < prop.width && h < prop.height
        x = radial_x(Float32(w), Float32(h), prop)
        y = radial_y(Float32(w), Float32(h), prop)
        out = h * prop.width + w + Int32(1)
        @inbounds dst[1, out] = interp_channel(src, Int32(1), y, x, prop)
        @inbounds dst[2, out] = interp_channel(src, Int32(2), y, x, prop)
        @inbounds dst[3, out] = interp_channel(src, Int32(3), y, x, prop)
    end
    return
end

function reference!(src, dst, prop::Properties)
    @inbounds for h in 0:(Int(prop.height) - 1), w in 0:(Int(prop.width) - 1)
        x = radial_x(Float32(w), Float32(h), prop)
        y = radial_y(Float32(w), Float32(h), prop)
        out = h * Int(prop.width) + w + 1
        dst[1, out] = interp_channel(src, Int32(1), y, x, prop)
        dst[2, out] = interp_channel(src, Int32(2), y, x, prop)
        dst[3, out] = interp_channel(src, Int32(3), y, x, prop)
    end
end

function main(args)
    if length(args) != 4
        println("Usage: main.jl <input image width> <input image height> <coefficient of distortion> <repeat>")
        return 1
    end
    width = parse(Int, args[1])
    height = parse(Int, args[2])
    k = parse(Float32, args[3])
    repeat = parse(Int, args[4])
    prop = make_properties(width, height, k)
    image_size = width * height

    src = Matrix{UInt8}(undef, 3, image_size)
    libc_srand(123)
    @inbounds for i in 1:image_size
        src[1, i] = UInt8(mod(libc_rand(), 256))
        src[2, i] = UInt8(mod(libc_rand(), 256))
        src[3, i] = UInt8(mod(libc_rand(), 256))
    end

    d_src = CuArray(src)
    d_dst = CUDA.zeros(UInt8, 3, image_size)
    threads = (16, 16)
    blocks = (width ÷ threads[1] + 1, height ÷ threads[2] + 1)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks distort_kernel!(d_src, d_dst, prop)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average kernel execution time: %f (ms)\n", elapsed * 1.0e-6 / repeat)

    dst = Array(d_dst)
    ref = similar(dst)
    reference!(src, ref, prop)
    ex = maximum(abs.(Int.(dst[1, :]) .- Int.(ref[1, :])))
    ey = maximum(abs.(Int.(dst[2, :]) .- Int.(ref[2, :])))
    ez = maximum(abs.(Int.(dst[3, :]) .- Int.(ref[3, :])))
    @printf("Max error of each channel: %d %d %d\n", ex, ey, ez)
    println((ex == 0 && ey == 0 && ez == 0) ? "PASS" : "FAIL")
    return (ex == 0 && ey == 0 && ez == 0) ? 0 : 1
end

exit(main(ARGS))
