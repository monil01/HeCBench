using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function row_jacc!(xi,yi,input,temp,w,h,scale,template)
    lid=(xi-1)%90;groupx=(xi-1)÷90;groupy=yi-1
    scratch=JACC.shared(template)
    x=groupx*64+lid-16;y=groupy
    if 0<=x<w
        @inbounds scratch[lid+1]=input[y*w+x+1]
    else
        scratch[lid+1]=UInt32(0)
    end
    JACC.sync_workgroup()
    if 0<=x<w && 16<=lid<80
        value=(0f0,0f0,0f0,0f0)
        for offset in lid-10:lid+10
            @inbounds value=add4(value,rgba(scratch[offset+1]))
        end
        @inbounds temp[y*w+x+1]=pack4(value,scale)
    end
    return
end
function col_jacc!(index,temp,out,w,h,scale)
    x=index-1
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
    template=CUDA.zeros(UInt32,90)
    row_spec=JACC.launch_spec(;threads=(90,1),blocks=(cld(Int(w),64),Int(h)),shmem_size=360,sync=false)
    col_spec=JACC.launch_spec(;threads=64,blocks=cld(Int(w),64),shmem_size=0,sync=false)
    ()->begin
        JACC.parallel_for(row_spec,(cld(Int(w),64)*90,Int(h)),row_jacc!,input,temp,w,h,scale,template)
        JACC.parallel_for(col_spec,Int(w),col_jacc!,temp,out,w,h,scale)
    end
end
box_study(factory)
