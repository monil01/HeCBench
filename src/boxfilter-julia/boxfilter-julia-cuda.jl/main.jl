using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function row_cuda!(input,temp,w,h,scale)
    lid=Int32(threadIdx().x)-1;groupx=Int32(blockIdx().x)-1;groupy=Int32(blockIdx().y)-1
    scratch=CuStaticSharedArray(UInt32,90)
    x=groupx*64+lid-16;y=groupy
    if 0<=x<w
        @inbounds scratch[lid+1]=input[y*w+x+1]
    else
        scratch[lid+1]=UInt32(0)
    end
    sync_threads()
    if 0<=x<w && 16<=lid<80
        value=(0f0,0f0,0f0,0f0)
        for offset in lid-10:lid+10
            @inbounds value=add4(value,rgba(scratch[offset+1]))
        end
        @inbounds temp[y*w+x+1]=pack4(value,scale)
    end
    return
end
function col_cuda!(temp,out,w,h,scale)
    x=(Int32(blockIdx().x)-1)*Int32(blockDim().x)+Int32(threadIdx().x)-1
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
    return
end
function factory(input,temp,out,w,h,scale)
    ()->begin
        @cuda threads=90 blocks=(cld(Int(w),64),Int(h)) row_cuda!(input,temp,w,h,scale)
        @cuda threads=64 blocks=cld(Int(w),64) col_cuda!(temp,out,w,h,scale)
    end
end
box_study(factory)
