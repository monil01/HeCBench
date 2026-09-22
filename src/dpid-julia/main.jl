using CUDA
using Printf

struct Params
    o_width::UInt32
    o_height::UInt32
    i_width::UInt32
    i_height::UInt32
    p_width::Float32
    p_height::Float32
    lambda::Float32
    repeat::UInt32
end

function usage()
    println("Usage: ", PROGRAM_FILE, " <output image width> <output image height> <lambda> <repeat>")
    exit(1)
end

function lcg_random_double(seed::Base.RefValue{UInt64})
    a = UInt64(2806196910506780709)
    seed[] = a * seed[] + UInt64(1)
    seed[] &= UInt64(0x7fffffffffffffff)
    return Float64(seed[]) / Float64(UInt64(1) << 63)
end

function init_input(i_width::Int, i_height::Int)
    seed = Ref(UInt64(123))
    input = Matrix{NTuple{3, UInt8}}(undef, i_width, i_height)
    @inbounds for y in 1:i_height, x in 1:i_width
        input[x, y] = (
            UInt8(floor(Int, 256 * lcg_random_double(seed))),
            UInt8(floor(Int, 256 * lcg_random_double(seed))),
            UInt8(floor(Int, 256 * lcg_random_double(seed))),
        )
    end
    return input
end

@inline function contribution(sx::Float32, ex::Float32, sy::Float32, ey::Float32,
                              f::Float32, x0::UInt32, y0::UInt32)
    xf = Float32(x0)
    yf = Float32(y0)
    if xf < sx
        f *= 1.0f0 - (sx - xf)
    end
    if xf + 1.0f0 > ex
        f *= 1.0f0 - ((xf + 1.0f0) - ex)
    end
    if yf < sy
        f *= 1.0f0 - (sy - yf)
    end
    if yf + 1.0f0 > ey
        f *= 1.0f0 - ((yf + 1.0f0) - ey)
    end
    return f
end

@inline function lambda_weight(lambda::Float32, dist::Float32)
    lambda == 0.0f0 && return 1.0f0
    lambda == 1.0f0 && return dist
    return dist ^ lambda
end

@inline function local_bounds(p::Params, px0::UInt32, py0::UInt32)
    sx = max(Float32(px0) * p.p_width, 0.0f0)
    ex = min(Float32(px0 + UInt32(1)) * p.p_width, Float32(p.i_width))
    sy = max(Float32(py0) * p.p_height, 0.0f0)
    ey = min(Float32(py0 + UInt32(1)) * p.p_height, Float32(p.i_height))
    sxr = UInt32(floor(sx))
    syr = UInt32(floor(sy))
    exr = UInt32(ceil(ex))
    eyr = UInt32(ceil(ey))
    return sx, ex, sy, ey, sxr, syr, exr, eyr
end

function guidance_cpu(input, p::Params)
    patches = Matrix{NTuple{3, UInt8}}(undef, Int(p.o_width), Int(p.o_height))
    @inbounds for py0 in UInt32(0):p.o_height-UInt32(1), px0 in UInt32(0):p.o_width-UInt32(1)
        sx, ex, sy, ey, sxr, syr, exr, eyr = local_bounds(p, px0, py0)
        cx = cy = cz = cw = 0.0f0
        for y0 in syr:eyr-UInt32(1), x0 in sxr:exr-UInt32(1)
            f = contribution(sx, ex, sy, ey, 1.0f0, x0, y0)
            pix = input[Int(x0) + 1, Int(y0) + 1]
            cx += Float32(pix[1]) * f
            cy += Float32(pix[2]) * f
            cz += Float32(pix[3]) * f
            cw += f
        end
        patches[Int(px0) + 1, Int(py0) + 1] = (UInt8(trunc(cx / cw)), UInt8(trunc(cy / cw)), UInt8(trunc(cz / cw)))
    end
    return patches
end

function calc_average(patches, p::Params, px0::UInt32, py0::UInt32)
    corner = 1.0f0
    edge = 2.0f0
    center = 4.0f0
    ax = ay = az = aw = 0.0f0
    function add_patch(qx0::UInt32, qy0::UInt32, w::Float32)
        pix = @inbounds patches[Int(qx0) + 1, Int(qy0) + 1]
        return Float32(pix[1]) * w, Float32(pix[2]) * w, Float32(pix[3]) * w, w
    end
    if py0 > 0
        if px0 > 0
            x, y, z, w = add_patch(px0 - UInt32(1), py0 - UInt32(1), corner); ax += x; ay += y; az += z; aw += w
        end
        x, y, z, w = add_patch(px0, py0 - UInt32(1), edge); ax += x; ay += y; az += z; aw += w
        if px0 + UInt32(1) < p.o_width
            x, y, z, w = add_patch(px0 + UInt32(1), py0 - UInt32(1), corner); ax += x; ay += y; az += z; aw += w
        end
    end
    if px0 > 0
        x, y, z, w = add_patch(px0 - UInt32(1), py0, edge); ax += x; ay += y; az += z; aw += w
    end
    x, y, z, w = add_patch(px0, py0, center); ax += x; ay += y; az += z; aw += w
    if px0 + UInt32(1) < p.o_width
        x, y, z, w = add_patch(px0 + UInt32(1), py0, edge); ax += x; ay += y; az += z; aw += w
    end
    if py0 + UInt32(1) < p.o_height
        if px0 > 0
            x, y, z, w = add_patch(px0 - UInt32(1), py0 + UInt32(1), corner); ax += x; ay += y; az += z; aw += w
        end
        x, y, z, w = add_patch(px0, py0 + UInt32(1), edge); ax += x; ay += y; az += z; aw += w
        if px0 + UInt32(1) < p.o_width
            x, y, z, w = add_patch(px0 + UInt32(1), py0 + UInt32(1), corner); ax += x; ay += y; az += z; aw += w
        end
    end
    return ax / aw, ay / aw, az / aw
end

function downsample_cpu(input, patches, p::Params)
    output = Matrix{NTuple{3, UInt8}}(undef, Int(p.o_width), Int(p.o_height))
    @inbounds for py0 in UInt32(0):p.o_height-UInt32(1), px0 in UInt32(0):p.o_width-UInt32(1)
        sx, ex, sy, ey, sxr, syr, exr, eyr = local_bounds(p, px0, py0)
        avgx, avgy, avgz = calc_average(patches, p, px0, py0)
        cx = cy = cz = cw = 0.0f0
        for y0 in syr:eyr-UInt32(1), x0 in sxr:exr-UInt32(1)
            pix = input[Int(x0) + 1, Int(y0) + 1]
            dx = avgx - Float32(pix[1])
            dy = avgy - Float32(pix[2])
            dz = avgz - Float32(pix[3])
            f = sqrt(dx * dx + dy * dy + dz * dz) / 441.6729559f0
            f = lambda_weight(p.lambda, f)
            f = contribution(sx, ex, sy, ey, f, x0, y0)
            cx += Float32(pix[1]) * f
            cy += Float32(pix[2]) * f
            cz += Float32(pix[3]) * f
            cw += f
        end
        if cw == 0.0f0
            output[Int(px0) + 1, Int(py0) + 1] = (UInt8(trunc(avgx)), UInt8(trunc(avgy)), UInt8(trunc(avgz)))
        else
            output[Int(px0) + 1, Int(py0) + 1] = (UInt8(trunc(cx / cw)), UInt8(trunc(cy / cw)), UInt8(trunc(cz / cw)))
        end
    end
    return output
end

function guidance_kernel!(input, patches, p::Params)
    idx0 = UInt32((blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x - UInt32(1))
    total = p.o_width * p.o_height
    if idx0 < total
        px0 = idx0 % p.o_width
        py0 = idx0 ÷ p.o_width
        sx, ex, sy, ey, sxr, syr, exr, eyr = local_bounds(p, px0, py0)
        cx = cy = cz = cw = 0.0f0
        y0 = syr
        while y0 < eyr
            x0 = sxr
            while x0 < exr
                f = contribution(sx, ex, sy, ey, 1.0f0, x0, y0)
                pix = @inbounds input[Int(x0) + 1, Int(y0) + 1]
                cx += Float32(pix[1]) * f
                cy += Float32(pix[2]) * f
                cz += Float32(pix[3]) * f
                cw += f
                x0 += UInt32(1)
            end
            y0 += UInt32(1)
        end
        @inbounds patches[Int(px0) + 1, Int(py0) + 1] =
            (UInt8(trunc(cx / cw)), UInt8(trunc(cy / cw)), UInt8(trunc(cz / cw)))
    end
    return
end

function add_patch_device(patches, p::Params, qx0::UInt32, qy0::UInt32,
                          w::Float32, ax::Float32, ay::Float32,
                          az::Float32, aw::Float32)
    pix = @inbounds patches[Int(qx0) + 1, Int(qy0) + 1]
    ax += Float32(pix[1]) * w
    ay += Float32(pix[2]) * w
    az += Float32(pix[3]) * w
    aw += w
    return ax, ay, az, aw
end

function downsample_kernel!(input, patches, output, p::Params)
    idx0 = UInt32((blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x - UInt32(1))
    total = p.o_width * p.o_height
    if idx0 < total
        px0 = idx0 % p.o_width
        py0 = idx0 ÷ p.o_width
        sx, ex, sy, ey, sxr, syr, exr, eyr = local_bounds(p, px0, py0)
        ax = ay = az = aw = 0.0f0
        if py0 > UInt32(0)
            if px0 > UInt32(0)
                ax, ay, az, aw = add_patch_device(patches, p, px0 - UInt32(1), py0 - UInt32(1), 1.0f0, ax, ay, az, aw)
            end
            ax, ay, az, aw = add_patch_device(patches, p, px0, py0 - UInt32(1), 2.0f0, ax, ay, az, aw)
            if px0 + UInt32(1) < p.o_width
                ax, ay, az, aw = add_patch_device(patches, p, px0 + UInt32(1), py0 - UInt32(1), 1.0f0, ax, ay, az, aw)
            end
        end
        if px0 > UInt32(0)
            ax, ay, az, aw = add_patch_device(patches, p, px0 - UInt32(1), py0, 2.0f0, ax, ay, az, aw)
        end
        ax, ay, az, aw = add_patch_device(patches, p, px0, py0, 4.0f0, ax, ay, az, aw)
        if px0 + UInt32(1) < p.o_width
            ax, ay, az, aw = add_patch_device(patches, p, px0 + UInt32(1), py0, 2.0f0, ax, ay, az, aw)
        end
        if py0 + UInt32(1) < p.o_height
            if px0 > UInt32(0)
                ax, ay, az, aw = add_patch_device(patches, p, px0 - UInt32(1), py0 + UInt32(1), 1.0f0, ax, ay, az, aw)
            end
            ax, ay, az, aw = add_patch_device(patches, p, px0, py0 + UInt32(1), 2.0f0, ax, ay, az, aw)
            if px0 + UInt32(1) < p.o_width
                ax, ay, az, aw = add_patch_device(patches, p, px0 + UInt32(1), py0 + UInt32(1), 1.0f0, ax, ay, az, aw)
            end
        end
        avgx = ax / aw
        avgy = ay / aw
        avgz = az / aw
        cx = cy = cz = cw = 0.0f0
        y0 = syr
        while y0 < eyr
            x0 = sxr
            while x0 < exr
                pix = @inbounds input[Int(x0) + 1, Int(y0) + 1]
                rx = Float32(pix[1])
                ry = Float32(pix[2])
                rz = Float32(pix[3])
                dx = avgx - rx
                dy = avgy - ry
                dz = avgz - rz
                f = sqrt(dx * dx + dy * dy + dz * dz) / 441.6729559f0
                f = lambda_weight(p.lambda, f)
                f = contribution(sx, ex, sy, ey, f, x0, y0)
                cx += rx * f
                cy += ry * f
                cz += rz * f
                cw += f
                x0 += UInt32(1)
            end
            y0 += UInt32(1)
        end
        if cw == 0.0f0
            @inbounds output[Int(px0) + 1, Int(py0) + 1] =
                (UInt8(trunc(avgx)), UInt8(trunc(avgy)), UInt8(trunc(avgz)))
        else
            @inbounds output[Int(px0) + 1, Int(py0) + 1] =
                (UInt8(trunc(cx / cw)), UInt8(trunc(cy / cw)), UInt8(trunc(cz / cw)))
        end
    end
    return
end

function run_downsampling(input, p::Params)
    d_input = CuArray(input)
    d_patches = CuArray{NTuple{3, UInt8}}(undef, Int(p.o_width), Int(p.o_height))
    d_output = similar(d_patches)
    threads = 128
    blocks = cld(Int(p.o_width) * Int(p.o_height), threads)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:Int(p.repeat)
        @cuda threads=threads blocks=blocks guidance_kernel!(d_input, d_patches, p)
        @cuda threads=threads blocks=blocks downsample_kernel!(d_input, d_patches, d_output, p)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    return Array(d_output), elapsed
end

function main()
    length(ARGS) == 4 || usage()
    o_width = parse(UInt32, ARGS[1])
    o_height = parse(UInt32, ARGS[2])
    lambda = parse(Float32, ARGS[3])
    repeat = parse(UInt32, ARGS[4])
    if o_width == 0 && o_height == 0
        println("only one dimension (width or height) can be 0!")
        exit(1)
    end

    i_width = UInt32(8192)
    i_height = UInt32(8192)
    if o_width == 0
        o_width = UInt32(round((Float64(o_height) / Float64(i_height)) * Float64(i_width)))
    end
    if o_height == 0
        o_height = UInt32(round((Float64(o_width) / Float64(i_width)) * Float64(i_height)))
    end
    p = Params(o_width, o_height, i_width, i_height,
               Float32(Float64(i_width) / Float64(o_width)),
               Float32(Float64(i_height) / Float64(o_height)),
               lambda, repeat)

    input = init_input(Int(i_width), Int(i_height))
    output, elapsed = run_downsampling(input, p)
    @printf("Average kernel execution time %f (s)\n", elapsed * 1.0e-9 / Float64(repeat))

    ref = downsample_cpu(input, guidance_cpu(input, p), p)
    mismatches = UInt32(0)
    maxdiff = 0
    @inbounds for i in eachindex(output)
        dx = abs(Int(output[i][1]) - Int(ref[i][1]))
        dy = abs(Int(output[i][2]) - Int(ref[i][2]))
        dz = abs(Int(output[i][3]) - Int(ref[i][3]))
        if dx > 0 || dy > 0 || dz > 0
            mismatches += UInt32(1)
        end
        maxdiff = max(maxdiff, dx, dy, dz)
    end
    total = UInt32(length(output))
    if mismatches == 0
        println("Verification PASS: GPU matches CPU reference exactly.")
        println("PASS")
    else
        @printf("Verification: %u / %u pixels differ (%.2f%%), max channel diff = %d\n",
                mismatches, total, 100.0 * Float64(mismatches) / Float64(total), maxdiff)
        if maxdiff <= 1
            println("  -> within rounding tolerance (maxDiff<=1, likely OK)")
            println("PASS")
        else
            println("  -> WARNING: differences exceed rounding tolerance")
            println("FAIL")
        end
    end
end

main()
