struct Properties
 K::Float32
 centerX::Float32
 centerY::Float32
 width::Int32
 height::Int32
 thresh::Float32
 xscale::Float32
 yscale::Float32
 xshift::Float32
 yshift::Float32
end
Base.zero(::Type{Properties})=Properties(0f0,0f0,0f0,Int32(0),Int32(0),0f0,0f0,0f0,0f0,0f0)
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

