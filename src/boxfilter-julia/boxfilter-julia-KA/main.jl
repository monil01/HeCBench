using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function row_ka!(input,temp,w,h,scale)
    lid=(@index(Local,Linear))-1
    gx,gy=@index(Group,NTuple);groupx=gx-1;groupy=gy-1
    scratch=@localmem UInt32 (90,)
    x=groupx*64+lid-16;y=groupy
    if 0<=x<w
        @inbounds scratch[lid+1]=input[y*w+x+1]
    else
        scratch[lid+1]=UInt32(0)
    end
    @synchronize
    if 0<=x<w && 16<=lid<80
        value=(0f0,0f0,0f0,0f0)
        for offset in lid-10:lid+10
            @inbounds value=add4(value,rgba(scratch[offset+1]))
        end
        @inbounds temp[y*w+x+1]=pack4(value,scale)
    end
end
@kernel function col_ka!(temp,out,w,h,scale)
    x=(@index(Global,Linear))-1
    if x<w
        @inbounds top=rgba(temp[x+1]);bottom=rgba(temp[(h-1)*w+x+1])
        value=scale4(top,10f0)
        for y in 0:10
            @inbounds value=add4(value,rgba(temp[y*w+x+1]))
        end
        # Native intentionally reproduced: no row0 output store here.
        for y in 1:10
            @inbounds value=sub4(add4(value,rgba(temp[(y+10)*w+x+1])),top)
            @inbounds out[y*w+x+1]=pack4(value,scale)
        end
        for y in 11:h-11
            @inbounds value=sub4(add4(value,rgba(temp[(y+10)*w+x+1])),rgba(temp[(y-11)*w+x+1]))
            @inbounds out[y*w+x+1]=pack4(value,scale)
        end
        for y in h-10:h-1
            @inbounds value=sub4(add4(value,bottom),rgba(temp[(y-11)*w+x+1]))
            @inbounds out[y*w+x+1]=pack4(value,scale)
        end
    end
end
function factory(input,temp,out,w,h,scale)
    ()->begin
        row_ka!(CUDABackend(),(90,1))(input,temp,w,h,scale;ndrange=(cld(Int(w),64)*90,Int(h)))
        col_ka!(CUDABackend(),64)(temp,out,w,h,scale;ndrange=Int(w))
    end
end
box_study(factory)
