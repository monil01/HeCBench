struct NoiseParameters
    ppu::Float32
    seed::Int32
    octaves::Int32
    lacunarity::Float32
    persistence::Float32
end
@inline smooth_noise(t)=t*t*t*muladd(t,muladd(t,6f0,-15f0),10f0)
@inline lerp_noise(a,b,t)=muladd(t,b-a,a)
@inline function noise_at(x,y,seed,hash,gx,gy)
    ix=unsafe_trunc(Int32,x);iy=unsafe_trunc(Int32,y)
    wx=x-Float32(ix);wy=y-Float32(iy)
    ix0=ix&Int32(255);iy0=iy&Int32(255)
    ix1=(ix0+Int32(1))&Int32(255);iy1=(iy0+Int32(1))&Int32(255)
    @inbounds h0=hash[ix0+1];h1=hash[ix1+1]
    @inbounds tl=(hash[((h0+iy0)&Int32(255))+1]+seed)%Int32(8)+1
    @inbounds tr=(hash[((h1+iy0)&Int32(255))+1]+seed)%Int32(8)+1
    @inbounds bl=(hash[((h0+iy1)&Int32(255))+1]+seed)%Int32(8)+1
    @inbounds br=(hash[((h1+iy1)&Int32(255))+1]+seed)%Int32(8)+1
    @inbounds dtl=muladd(gx[tl],wx,gy[tl]*wy)
    @inbounds dtr=muladd(gx[tr],wx-1f0,gy[tr]*wy)
    @inbounds dbl=muladd(gx[bl],wx,gy[bl]*(wy-1f0))
    @inbounds dbr=muladd(gx[br],wx-1f0,gy[br]*(wy-1f0))
    tx=smooth_noise(wx);ty=smooth_noise(wy)
    v=lerp_noise(lerp_noise(dtl,dbl,ty),lerp_noise(dtr,dbr,ty),tx)
    Float32((Float64(v)+1.0)*0.5)
end
@inline function pixel_noise!(px,py,ystart,height,p,hash,gx,gy,out)
    if px<Int32(61440) && py<ystart+height
        x=Float32(px)/p.ppu;y=Float32(py)/p.ppu
        frequency=1f0;amplitude=1f0;range=1f0
        value=noise_at(x*frequency,y*frequency,p.seed,hash,gx,gy)
        for octave in Int32(1):p.octaves-Int32(1)
            frequency*=p.lacunarity;amplitude*=p.persistence;range+=amplitude
            value=muladd(amplitude,noise_at(x*frequency,y*frequency,p.seed,hash,gx,gy),value)
        end
        byte=unsafe_trunc(UInt8,(value/range)*255f0)
        base=4*(Int64(py)*61440+Int64(px))+1
        @inbounds out[base]=byte;out[base+1]=byte;out[base+2]=byte;out[base+3]=0xff
    end
    nothing
end
@inline function echo_noise!(out,p,height,ns)
    words=(reinterpret(UInt32,p.ppu),reinterpret(UInt32,p.seed),reinterpret(UInt32,p.octaves),reinterpret(UInt32,p.lacunarity),reinterpret(UInt32,p.persistence),UInt32(61440),UInt32(height),UInt32(ns),UInt32(height÷ns))
    for j in 1:9, k in 0:3
        @inbounds out[4*(j-1)+k+1]=unsafe_trunc(UInt8,words[j]>>(8*k))
    end
    nothing
end
