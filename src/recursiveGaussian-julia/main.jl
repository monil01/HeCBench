using CUDA
using Printf

struct GaussParms
    nsigma::Float32
    alpha::Float32
    ema::Float32
    ema2::Float32
    b1::Float32
    b2::Float32
    a0::Float32
    a1::Float32
    a2::Float32
    a3::Float32
    coefp::Float32
    coefn::Float32
end

@inline function chan(p::UInt32, shift::UInt32)
    return Float32((p >> shift) & UInt32(0xff))
end

@inline function pack_rgba(r::Float32, g::Float32, b::Float32, a::Float32)
    rr = UInt32(max(Int32(trunc(r)), Int32(0))) & UInt32(0xff)
    gg = UInt32(max(Int32(trunc(g)), Int32(0))) & UInt32(0xff)
    bb = UInt32(max(Int32(trunc(b)), Int32(0))) & UInt32(0xff)
    aa = UInt32(max(Int32(trunc(a)), Int32(0))) & UInt32(0xff)
    return rr | (gg << UInt32(8)) | (bb << UInt32(16)) | (aa << UInt32(24))
end

function preprocess_gauss_parms(sigma::Float32, order::Int32)
    alpha = 1.695f0 / sigma
    ema = exp(-alpha)
    ema2 = exp(-2.0f0 * alpha)
    b1 = -2.0f0 * ema
    b2 = ema2
    a0 = 0f0
    a1 = 0f0
    a2 = 0f0
    a3 = 0f0
    if order == 0
        k = (1.0f0 - ema) * (1.0f0 - ema) / (1.0f0 + (2.0f0 * alpha * ema) - ema2)
        a0 = k
        a1 = k * (alpha - 1.0f0) * ema
        a2 = k * (alpha + 1.0f0) * ema
        a3 = -k * ema2
    elseif order == 1
        a0 = (1.0f0 - ema) * (1.0f0 - ema)
        a2 = -a0
    elseif order == 2
        ea = exp(-alpha)
        k = -(ema2 - 1.0f0) / (2.0f0 * alpha * ema)
        kn = -2.0f0 * (-1.0f0 + (3.0f0 * ea) - (3.0f0 * ea * ea) + (ea * ea * ea))
        kn /= ((3.0f0 * ea) + 1.0f0 + (3.0f0 * ea * ea) + (ea * ea * ea))
        a0 = kn
        a1 = -kn * (1.0f0 + (k * alpha)) * ema
        a2 = kn * (1.0f0 - (k * alpha)) * ema
        a3 = -kn * ema2
    end
    coefp = (a0 + a1) / (1.0f0 + b1 + b2)
    coefn = (a2 + a3) / (1.0f0 + b1 + b2)
    return GaussParms(sigma, alpha, ema, ema2, b1, b2, a0, a1, a2, a3, coefp, coefn)
end

function load_ppm4ub(path::String)
    bytes = read(path)
    pos = 1
    function next_token()
        while pos <= length(bytes) && bytes[pos] in UInt8[0x20, 0x0a, 0x0d, 0x09]
            pos += 1
        end
        if bytes[pos] == UInt8('#')
            while pos <= length(bytes) && bytes[pos] != UInt8('\n')
                pos += 1
            end
            return next_token()
        end
        start = pos
        while pos <= length(bytes) && !(bytes[pos] in UInt8[0x20, 0x0a, 0x0d, 0x09])
            pos += 1
        end
        return String(bytes[start:(pos - 1)])
    end
    magic = next_token()
    width = parse(Int, next_token())
    height = parse(Int, next_token())
    maxval = parse(Int, next_token())
    if magic != "P6" || maxval != 255
        error("unsupported PPM")
    end
    while pos <= length(bytes) && bytes[pos] in UInt8[0x20, 0x0a, 0x0d, 0x09]
        pos += 1
    end
    out = Vector{UInt32}(undef, width * height)
    p = pos
    @inbounds for i in eachindex(out)
        r = UInt32(bytes[p])
        g = UInt32(bytes[p + 1])
        b = UInt32(bytes[p + 2])
        out[i] = r | (g << UInt32(8)) | (b << UInt32(16))
        p += 3
    end
    return out, width, height
end

function recursive_kernel!(input, output, width::Int32, height::Int32, gp::GaussParms)
    x0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if x0 >= width
        return
    end

    first = @inbounds input[x0 + Int32(1)]
    xr = chan(first, UInt32(0)); xg = chan(first, UInt32(8)); xb = chan(first, UInt32(16)); xa = chan(first, UInt32(24))
    ybr = xr * gp.coefp; ybg = xg * gp.coefp; ybb = xb * gp.coefp; yba = xa * gp.coefp
    ypr = ybr; ypg = ybg; ypb = ybb; ypa = yba
    xpr = xr; xpg = xg; xpb = xb; xpa = xa

    for y0 in Int32(0):(height - Int32(1))
        idx = y0 * width + x0 + Int32(1)
        pix = @inbounds input[idx]
        xr = chan(pix, UInt32(0)); xg = chan(pix, UInt32(8)); xb = chan(pix, UInt32(16)); xa = chan(pix, UInt32(24))
        yr = (xr * gp.a0) + (xpr * gp.a1) - (ypr * gp.b1) - (ybr * gp.b2)
        yg = (xg * gp.a0) + (xpg * gp.a1) - (ypg * gp.b1) - (ybg * gp.b2)
        yb = (xb * gp.a0) + (xpb * gp.a1) - (ypb * gp.b1) - (ybb * gp.b2)
        ya = (xa * gp.a0) + (xpa * gp.a1) - (ypa * gp.b1) - (yba * gp.b2)
        @inbounds output[idx] = pack_rgba(yr, yg, yb, ya)
        xpr = xr; xpg = xg; xpb = xb; xpa = xa
        ybr = ypr; ybg = ypg; ybb = ypb; yba = ypa
        ypr = yr; ypg = yg; ypb = yb; ypa = ya
    end

    last = @inbounds input[(height - Int32(1)) * width + x0 + Int32(1)]
    xnr = chan(last, UInt32(0)); xng = chan(last, UInt32(8)); xnb = chan(last, UInt32(16)); xna = chan(last, UInt32(24))
    xar = xnr; xag = xng; xab = xnb; xaa = xna
    ynr = xnr * gp.coefn; yng = xng * gp.coefn; ynb = xnb * gp.coefn; yna = xna * gp.coefn
    yar = ynr; yag = yng; yab = ynb; yaa = yna

    for y0 in (height - Int32(1)):-Int32(1):Int32(0)
        idx = y0 * width + x0 + Int32(1)
        pix = @inbounds input[idx]
        xr = chan(pix, UInt32(0)); xg = chan(pix, UInt32(8)); xb = chan(pix, UInt32(16)); xa = chan(pix, UInt32(24))
        yr = (xnr * gp.a2) + (xar * gp.a3) - (ynr * gp.b1) - (yar * gp.b2)
        yg = (xng * gp.a2) + (xag * gp.a3) - (yng * gp.b1) - (yag * gp.b2)
        yb = (xnb * gp.a2) + (xab * gp.a3) - (ynb * gp.b1) - (yab * gp.b2)
        ya = (xna * gp.a2) + (xaa * gp.a3) - (yna * gp.b1) - (yaa * gp.b2)
        old = @inbounds output[idx]
        @inbounds output[idx] = pack_rgba(chan(old, UInt32(0)) + yr, chan(old, UInt32(8)) + yg,
                                          chan(old, UInt32(16)) + yb, chan(old, UInt32(24)) + ya)
        xar = xnr; xag = xng; xab = xnb; xaa = xna
        xnr = xr; xng = xg; xnb = xb; xna = xa
        yar = ynr; yag = yng; yab = ynb; yaa = yna
        ynr = yr; yng = yg; ynb = yb; yna = ya
    end
    return
end

function transpose_kernel!(input, output, width::Int32, height::Int32)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if x < width && y < height
        @inbounds output[x * height + y + Int32(1)] = input[y * width + x + Int32(1)]
    end
    return
end

function gpu_filter(input::Vector{UInt32}, width::Int, height::Int, gp::GaussParms, repeat::Int)
    n = width * height
    d_in = CuArray(input)
    d_tmp = CUDA.zeros(UInt32, n)
    d_out = CUDA.zeros(UInt32, n)
    threads_g = 256
    blocks_g1 = cld(width, threads_g)
    blocks_g2 = cld(height, threads_g)
    threads_t = (16, 16)
    blocks_t1 = (cld(width, 16), cld(height, 16))
    blocks_t2 = (cld(height, 16), cld(width, 16))

    function run_once()
        @cuda threads=threads_g blocks=blocks_g1 recursive_kernel!(d_in, d_tmp, Int32(width), Int32(height), gp)
        @cuda threads=threads_t blocks=blocks_t1 transpose_kernel!(d_tmp, d_out, Int32(width), Int32(height))
        @cuda threads=threads_g blocks=blocks_g2 recursive_kernel!(d_out, d_tmp, Int32(height), Int32(width), gp)
        @cuda threads=threads_t blocks=blocks_t2 transpose_kernel!(d_tmp, d_out, Int32(height), Int32(width))
    end

    run_once()
    CUDA.synchronize()
    total = 0
    for _ in 1:repeat
        start = time_ns()
        run_once()
        CUDA.synchronize()
        total += time_ns() - start
    end
    return Array(d_out), total
end

function host_filter(input::Vector{UInt32}, width::Int, height::Int, gp::GaussParms)
    tmp = similar(input)
    out = similar(input)
    recursive_host!(input, tmp, width, height, gp)
    transpose_host!(tmp, out, width, height)
    recursive_host!(out, tmp, height, width, gp)
    transpose_host!(tmp, out, height, width)
    return out
end

function recursive_host!(input, output, width::Int, height::Int, gp::GaussParms)
    @inbounds for x0 in 0:(width - 1)
        first = input[x0 + 1]
        xpr = chan(first, UInt32(0)); xpg = chan(first, UInt32(8)); xpb = chan(first, UInt32(16)); xpa = chan(first, UInt32(24))
        ybr = xpr * gp.coefp; ybg = xpg * gp.coefp; ybb = xpb * gp.coefp; yba = xpa * gp.coefp
        ypr = ybr; ypg = ybg; ypb = ybb; ypa = yba
        for y0 in 0:(height - 1)
            idx = y0 * width + x0 + 1
            pix = input[idx]
            xr = chan(pix, UInt32(0)); xg = chan(pix, UInt32(8)); xb = chan(pix, UInt32(16)); xa = chan(pix, UInt32(24))
            yr = (xr * gp.a0) + (xpr * gp.a1) - (ypr * gp.b1) - (ybr * gp.b2)
            yg = (xg * gp.a0) + (xpg * gp.a1) - (ypg * gp.b1) - (ybg * gp.b2)
            yb = (xb * gp.a0) + (xpb * gp.a1) - (ypb * gp.b1) - (ybb * gp.b2)
            ya = (xa * gp.a0) + (xpa * gp.a1) - (ypa * gp.b1) - (yba * gp.b2)
            output[idx] = pack_rgba(yr, yg, yb, ya)
            xpr = xr; xpg = xg; xpb = xb; xpa = xa
            ybr = ypr; ybg = ypg; ybb = ypb; yba = ypa
            ypr = yr; ypg = yg; ypb = yb; ypa = ya
        end
        last = input[(height - 1) * width + x0 + 1]
        xnr = chan(last, UInt32(0)); xng = chan(last, UInt32(8)); xnb = chan(last, UInt32(16)); xna = chan(last, UInt32(24))
        xar = xnr; xag = xng; xab = xnb; xaa = xna
        ynr = xnr * gp.coefn; yng = xng * gp.coefn; ynb = xnb * gp.coefn; yna = xna * gp.coefn
        yar = ynr; yag = yng; yab = ynb; yaa = yna
        for y0 in (height - 1):-1:0
            idx = y0 * width + x0 + 1
            pix = input[idx]
            xr = chan(pix, UInt32(0)); xg = chan(pix, UInt32(8)); xb = chan(pix, UInt32(16)); xa = chan(pix, UInt32(24))
            yr = (xnr * gp.a2) + (xar * gp.a3) - (ynr * gp.b1) - (yar * gp.b2)
            yg = (xng * gp.a2) + (xag * gp.a3) - (yng * gp.b1) - (yag * gp.b2)
            yb = (xnb * gp.a2) + (xab * gp.a3) - (ynb * gp.b1) - (yab * gp.b2)
            ya = (xna * gp.a2) + (xaa * gp.a3) - (yna * gp.b1) - (yaa * gp.b2)
            old = output[idx]
            output[idx] = pack_rgba(chan(old, UInt32(0)) + yr, chan(old, UInt32(8)) + yg,
                                    chan(old, UInt32(16)) + yb, chan(old, UInt32(24)) + ya)
            xar = xnr; xag = xng; xab = xnb; xaa = xna
            xnr = xr; xng = xg; xnb = xb; xna = xa
            yar = ynr; yag = yng; yab = ynb; yaa = yna
            ynr = yr; yng = yg; ynb = yb; yna = ya
        end
    end
end

function transpose_host!(input, output, width::Int, height::Int)
    @inbounds for y in 0:(height - 1), x in 0:(width - 1)
        output[x * height + y + 1] = input[y * width + x + 1]
    end
end

function compare_uint(reference, data)
    mismatches = 0
    @inbounds for i in eachindex(reference)
        if abs(Int64(reference[i]) - Int64(data[i])) > 1
            mismatches += 1
        end
    end
    return mismatches
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <path to image> <repeat>")
        return 1
    end
    image_path = args[1]
    if !isfile(image_path)
        sibling_path = joinpath(@__DIR__, "..", "recursiveGaussian-cuda", image_path)
        image_path = isfile(sibling_path) ? sibling_path : image_path
    end
    input, width, height = load_ppm4ub(image_path)
    @printf("Image Width = %i, Height = %i, bpp = %lu\n\n", width, height, 32)
    if width > 1920 || height > 1080
        println("Error: Image Dimensions exceed the maximum values")
        return 1
    end
    repeat = parse(Int, args[2])
    println("Allocate Host Image Buffers...")
    gp = preprocess_gauss_parms(10.0f0, Int32(0))
    println("\nRunning GPUGaussianFilterRGBA for $repeat cycles...\n")
    output, elapsed = gpu_filter(input, width, height, gp, repeat)
    @printf("Average execution time of kernels: %f (s)\n", elapsed * 1.0e-9 / repeat)
    golden = host_filter(input, width, height, gp)
    println("Comparing GPU Result to CPU Result...")
    mismatches = compare_uint(golden, output)
    @printf("%f(%%) of bytes mismatched (count=%d)\n",
            100.0 * mismatches / length(output), mismatches)
    matched = mismatches <= max(1, length(output) ÷ 100)
    @printf("\nGPU Result %s CPU Result within tolerance...\n", matched ? "matches" : "DOESN'T match")
    println(matched ? "PASS" : "FAIL")
    return matched ? 0 : 1
end

exit(main(ARGS))
