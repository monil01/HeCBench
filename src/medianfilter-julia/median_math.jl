# Original generic CUDA.jl binary-search median function, unchanged algorithm.
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

