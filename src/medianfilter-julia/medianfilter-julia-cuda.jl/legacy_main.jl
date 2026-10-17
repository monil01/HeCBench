using CUDA
using Printf

function load_ppm4ub(path::String)
    bytes = read(path)
    pos = 1
    function token()
        while pos <= length(bytes) && bytes[pos] in UInt8[0x20, 0x0a, 0x0d, 0x09]
            pos += 1
        end
        if bytes[pos] == UInt8('#')
            while pos <= length(bytes) && bytes[pos] != UInt8('\n')
                pos += 1
            end
            return token()
        end
        start = pos
        while pos <= length(bytes) && !(bytes[pos] in UInt8[0x20, 0x0a, 0x0d, 0x09])
            pos += 1
        end
        return String(bytes[start:(pos - 1)])
    end
    magic = token()
    width = parse(Int, token())
    height = parse(Int, token())
    maxval = parse(Int, token())
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

@inline function channel(p::UInt32, s::UInt32)
    return Float32((p >> s) & UInt32(0xff))
end

@inline function median_pixel(input, width::Int32, height::Int32, x::Int32, y::Int32)
    mr = 128.0f0; mg = 128.0f0; mb = 128.0f0
    minr = 0.0f0; ming = 0.0f0; minb = 0.0f0
    maxr = 255.0f0; maxg = 255.0f0; maxb = 255.0f0
    for _ in 1:8
        hr = UInt32(0); hg = UInt32(0); hb = UInt32(0)
        for dy in Int32(-1):Int32(1), dx in Int32(-1):Int32(1)
            xx = x + dx
            yy = y + dy
            pix = UInt32(0)
            if xx >= 0 && xx < width && yy >= 0 && yy < height
                pix = @inbounds input[yy * width + xx + Int32(1)]
            end
            hr += mr < channel(pix, UInt32(0)) ? UInt32(1) : UInt32(0)
            hg += mg < channel(pix, UInt32(8)) ? UInt32(1) : UInt32(0)
            hb += mb < channel(pix, UInt32(16)) ? UInt32(1) : UInt32(0)
        end
        if hr > 4; minr = mr else maxr = mr end
        if hg > 4; ming = mg else maxg = mg end
        if hb > 4; minb = mb else maxb = mb end
        mr = 0.5f0 * (maxr + minr)
        mg = 0.5f0 * (maxg + ming)
        mb = 0.5f0 * (maxb + minb)
    end
    r = UInt32(trunc(Int32, mr + 0.5f0)) & UInt32(0xff)
    g = (UInt32(trunc(Int32, mg + 0.5f0)) << UInt32(8)) & UInt32(0x0000ff00)
    b = (UInt32(trunc(Int32, mb + 0.5f0)) << UInt32(16)) & UInt32(0x00ff0000)
    return r | g | b
end

function median_kernel!(input, output, width::Int32, height::Int32)
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if x < width && y < height
        @inbounds output[y * width + x + Int32(1)] = median_pixel(input, width, height, x, y)
    end
    return
end

function median_host(input, width, height)
    out = Vector{UInt32}(undef, width * height)
    @inbounds for y in 0:(height - 1), x in 0:(width - 1)
        out[y * width + x + 1] = median_pixel(input, Int32(width), Int32(height), Int32(x), Int32(y))
    end
    return out
end

function median_gpu(input, width, height, repeat)
    d_in = CuArray(input)
    d_out = CUDA.zeros(UInt32, width * height)
    threads = (16, 4)
    blocks = (cld(width, 16), cld(height, 4))
    @cuda threads=threads blocks=blocks median_kernel!(d_in, d_out, Int32(width), Int32(height))
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks median_kernel!(d_in, d_out, Int32(width), Int32(height))
    end
    CUDA.synchronize()
    elapsed = time_ns() - t0
    return Array(d_out), elapsed
end

function compare_uint(golden, output)
    bad = 0
    @inbounds for i in eachindex(golden)
        g = golden[i]; o = output[i]
        dr = abs(Int((g >> UInt32(0)) & UInt32(0xff)) - Int((o >> UInt32(0)) & UInt32(0xff)))
        dg = abs(Int((g >> UInt32(8)) & UInt32(0xff)) - Int((o >> UInt32(8)) & UInt32(0xff)))
        db = abs(Int((g >> UInt32(16)) & UInt32(0xff)) - Int((o >> UInt32(16)) & UInt32(0xff)))
        bad += (dr > 1 || dg > 1 || db > 1) ? 1 : 0
    end
    return bad <= max(1, length(golden) ÷ 10000)
end

function main(args)
    if length(args) != 2
        println("Usage: main.jl <image file> <repeat>")
        return 1
    end
    input, width, height = load_ppm4ub(args[1])
    repeat = parse(Int, args[2])
    @printf("Image File\t = %s\nImage Dimensions = %u w x %u h x %lu bpp\n\n", args[1], width, height, sizeof(UInt32) << 3)
    println("\nRunning MedianFilterGPU for $repeat cycles...\n")
    output, elapsed = median_gpu(input, width, height, repeat)
    @printf("Average kernel execution time: %f (s)\n\n", elapsed * 1.0e-9 / repeat)
    golden = median_host(input, width, height)
    println("Comparing GPU Result to CPU Result...")
    ok = compare_uint(golden, output)
    @printf("\nGPU Result %s CPU Result within tolerance...\n", ok ? "matches" : "DOESN'T match")
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
