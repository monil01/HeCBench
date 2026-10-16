using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const RY = Int32(15)
const YG = Int32(6)
const GC_SEG = Int32(4)
const CB = Int32(11)
const BM = Int32(13)
const MR = Int32(6)
const MAXCOLS = Int32(55)

@inline function colorwheel_value(k::Int32, channel::Int32)
    if k < RY
        i = k
        r = Int32(255)
        g = Int32(255) * i ÷ RY
        b = Int32(0)
    elseif k < RY + YG
        i = k - RY
        r = Int32(255) - Int32(255) * i ÷ YG
        g = Int32(255)
        b = Int32(0)
    elseif k < RY + YG + GC_SEG
        i = k - RY - YG
        r = Int32(0)
        g = Int32(255)
        b = Int32(255) * i ÷ GC_SEG
    elseif k < RY + YG + GC_SEG + CB
        i = k - RY - YG - GC_SEG
        r = Int32(0)
        g = Int32(255) - Int32(255) * i ÷ CB
        b = Int32(255)
    elseif k < RY + YG + GC_SEG + CB + BM
        i = k - RY - YG - GC_SEG - CB
        r = Int32(255) * i ÷ BM
        g = Int32(0)
        b = Int32(255)
    else
        i = k - RY - YG - GC_SEG - CB - BM
        r = Int32(255)
        g = Int32(0)
        b = Int32(255) - Int32(255) * i ÷ MR
    end

    return channel == Int32(0) ? r : (channel == Int32(1) ? g : b)
end

@inline function compute_color_channel(fx::Float32, fy::Float32, out_channel::Int32)
    rad = sqrt(fx * fx + fy * fy)
    a = atan(-fy, -fx) / Float32(pi)
    fk = (a + 1.0f0) * 0.5f0 * Float32(MAXCOLS - Int32(1))
    k0 = Int32(trunc(fk))
    k1 = k0 + Int32(1)
    if k1 == MAXCOLS
        k1 = Int32(0)
    end
    f = fk - Float32(k0)

    cw_channel = Int32(2) - out_channel
    col0 = Float32(colorwheel_value(k0, cw_channel)) / 255.0f0
    col1 = Float32(colorwheel_value(k1, cw_channel)) / 255.0f0
    col = (1.0f0 - f) * col0 + f * col1
    if rad <= 1.0f0
        col = 1.0f0 - rad * (1.0f0 - col)
    else
        col *= 0.75f0
    end
    return UInt8(trunc(Int32, 255.0f0 * col))
end

function portable_color!(ix,iy,pix,size::Int32,half_size::Int32,range::Float32,truerange::Float32)
    x=Int32(ix)-Int32(1);y=Int32(iy)-Int32(1)
    if y<size && x<size && x!=half_size && y!=half_size
        fx=Float32(x)/Float32(half_size)*range-range;fy=Float32(y)/Float32(half_size)*range-range;idx=(y*size+x)*Int32(3)
        @inbounds begin
            pix[idx+Int32(1)]=compute_color_channel(fx/truerange,fy/truerange,Int32(0))
            pix[idx+Int32(2)]=compute_color_channel(fx/truerange,fy/truerange,Int32(1))
            pix[idx+Int32(3)]=compute_color_channel(fx/truerange,fy/truerange,Int32(2))
        end
    end
    nothing
end
function study_prepare(pix,size,half,range,truerange)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(32,32),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_color!,spec,(size,size),pix,Int32(size),Int32(half),range,truerange)
end
exit(study_main(study_prepare))
