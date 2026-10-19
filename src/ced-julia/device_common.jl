# Source-grounded logical 16x16 tile, row-major18x18 shared array.
# Guarded image storage starts at index17; original image position is zero-based.
@inline function load_tile!(tile,data,lr,lc,gr,gc,cols)
    pos=gr*cols+gc; center=lr*18+lc+1
    @inbounds tile[center]=Int32(data[pos+17])
    if lr==1
        @inbounds tile[lc+1]=Int32(data[pos-cols+17])
        if lc==1
            @inbounds tile[1]=Int32(data[pos-cols-1+17])
        elseif lc==16
            @inbounds tile[18]=Int32(data[pos-cols+1+17])
        end
    elseif lr==16
        @inbounds tile[17*18+lc+1]=Int32(data[pos+cols+17])
        if lc==1
            @inbounds tile[17*18+1]=Int32(data[pos+cols-1+17])
        elseif lc==16
            @inbounds tile[18*18]=Int32(data[pos+cols+1+17])
        end
    end
    if lc==1
        @inbounds tile[lr*18+1]=Int32(data[pos-1+17])
    elseif lc==16
        @inbounds tile[lr*18+18]=Int32(data[pos+1+17])
    end
    return nothing
end
@inline function gaussian_pixel!(tile,out,coeff,lr,lc,gr,gc,cols)
    total=Int32(0)
    @inbounds for y=0:2,x=0:2
        # C++ compound-assignment converts to Int32 after EACH contribution.
        total=trunc(Int32,Float32(total)+coeff[y*3+x+1]*Float32(tile[(lr+y-1)*18+lc+x]))
    end
    @inbounds out[gr*cols+gc+17]=UInt8(clamp(total,Int32(0),Int32(255)))
    return nothing
end
@inline function sobel_pixel!(tile,out,theta,sobx,soby,lr,lc,gr,gc,cols)
    sx=0.0f0;sy=0.0f0
    @inbounds for y=0:2,x=0:2
        value=tile[(lr+y-1)*18+lc+x]
        sx+=Float32(sobx[y*3+x+1]*value);sy+=Float32(soby[y*3+x+1]*value)
    end
    p=gr*cols+gc+17
    @inbounds out[p]=UInt8(clamp(trunc(Int32,hypot(sx,sy)),Int32(0),Int32(255)))
    angle=atan(sy,sx);pi32=3.14159265f0
    if angle<0.0f0
        angle=rem(angle+2.0f0*pi32,2.0f0*pi32)
    end
    direction=UInt8(0)
    if angle<=pi32/8.0f0
        direction=UInt8(0)
    elseif angle<=3.0f0*pi32/8.0f0
        direction=UInt8(45)
    elseif angle<=5.0f0*pi32/8.0f0
        direction=UInt8(90)
    elseif angle<=7.0f0*pi32/8.0f0
        direction=UInt8(135)
    elseif angle<=9.0f0*pi32/8.0f0
        direction=UInt8(0)
    elseif angle<=11.0f0*pi32/8.0f0
        direction=UInt8(45)
    elseif angle<=13.0f0*pi32/8.0f0
        direction=UInt8(90)
    elseif angle<=15.0f0*pi32/8.0f0
        direction=UInt8(135)
    end
    @inbounds theta[p]=direction
    return nothing
end
@inline function nms_pixel!(tile,out,theta,lr,lc,gr,gc,cols)
    p=gr*cols+gc+17;c=lr*18+lc+1
    @inbounds begin
        value=tile[c];dir=theta[p]
        delta=dir==0 ? 1 : dir==45 ? 17 : dir==90 ? 18 : 19
        out[p]=(value>tile[c-delta] && value>tile[c+delta]) ? UInt8(value) : UInt8(0)
    end
    return nothing
end
@inline function threshold_pixel!(data,out,gr,gc,cols)
    p=gr*cols+gc+17
    # Source high70/low10/median40 reduces to this point predicate.
    @inbounds out[p]=data[p]>=40 ? UInt8(255) : UInt8(0)
    return nothing
end
