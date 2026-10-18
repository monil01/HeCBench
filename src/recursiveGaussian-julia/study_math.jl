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

@inline function recursive_body!(x0::Int32,input, output, width::Int32, height::Int32, gp::GaussParms)
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

