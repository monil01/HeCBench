function interp_device(d1::Int32, d2::Int32, f, x::Float32, y::Float32, z::Float32)
    ix = Int32(floor(x)); dx1 = x - ix; dx2 = 1f0 - dx1
    iy = Int32(floor(y)); dy1 = y - iy; dy2 = 1f0 - dy1
    iz = Int32(floor(z)); dz1 = z - iz; dz2 = 1f0 - dz1
    base = ix + d1 * (iy - Int32(1) + d2 * (iz - Int32(1)))
    k222 = Float32(f[base]); k122 = Float32(f[base + Int32(1)])
    k212 = Float32(f[base + d1]); k112 = Float32(f[base + d1 + Int32(1)])
    base += d1 * d2
    k221 = Float32(f[base]); k121 = Float32(f[base + Int32(1)])
    k211 = Float32(f[base + d1]); k111 = Float32(f[base + d1 + Int32(1)])
    xy0=muladd(muladd(k222,dx2,k122*dx1),dy2,muladd(k212,dx2,k112*dx1)*dy1)
    xy1=muladd(muladd(k221,dx2,k121*dx1),dy2,muladd(k211,dx2,k111*dx1)*dy1)
    return muladd(xy0,dz2,xy1*dz1)
end

@kernel function spm_ka!(M, data_size::Int32, g, f, dg1::Int32, dg2::Int32, dg3::Int32,
                     df1::Int32, df2::Int32, df3::Int32, ivf, ivg, thresh)
    idx0 = Int32(@index(Global,Linear)-1)
    stride = Int32(128*256)
    x_datasize = dg1 - Int32(2)
    y_datasize = dg2 - Int32(2)
    i = idx0
    while i < data_size
        ii = Float32(i)
        xx = Float32(i % x_datasize) + 1f0
        yy = Float32((i ÷ x_datasize) % y_datasize) + 1f0
        zz = floor(ii / Float32(x_datasize)) / Float32(y_datasize) + 1f0
        r = RAN[(i % Int32(97)) + Int32(1)]
        rx = xx + r; ry = yy + r; rz = zz + r
        xp = muladd(M[9],rz,muladd(M[1],rx,M[5]*ry))+M[13]
        yp = muladd(M[10],rz,muladd(M[2],rx,M[6]*ry))+M[14]
        zp = muladd(M[11],rz,muladd(M[3],rx,M[7]*ry))+M[15]
        out = i + Int32(1)
        if zp >= 1f0 && zp < Float32(df3) && yp >= 1f0 && yp < Float32(df2) && xp >= 1f0 && xp < Float32(df1)
            ivf[out] = UInt8(floor(interp_device(df1, df2, f, xp, yp, zp) + 0.5f0))
            ivg[out] = UInt8(floor(interp_device(dg1, dg2, g, rx, ry, rz) + 0.5f0))
            thresh[out] = true
        else
            ivf[out] = UInt8(0)
            ivg[out] = UInt8(0)
            thresh[out] = false
        end
        i += stride
    end
end
