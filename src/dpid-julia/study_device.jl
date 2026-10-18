# AdaptedfromNicolasWeber/SandraC.Amend2016source; BSD3-Clause LICENSE at../dpid-cuda/LICENSE.
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


function guidance_lane!(idx0::UInt32,input, patches, p::Params)
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

function downsample_lane!(idx0::UInt32,input, patches, output, p::Params)
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

