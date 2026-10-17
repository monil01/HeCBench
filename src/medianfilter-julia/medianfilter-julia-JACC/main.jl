using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function median_tiled!(x,y,input,output,W,H,template)
    lx=(Int32(x)-Int32(1))%Int32(16)+Int32(1);ly=(Int32(y)-Int32(1))%Int32(4)+Int32(1);bx=(Int32(x)-Int32(1))÷Int32(16);by=(Int32(y)-Int32(1))÷Int32(4);lid=lx+(ly-Int32(1))*Int32(16);tile=JACC.shared(template)
    for j in lid:Int32(64):Int32(108)
        tx=(j-Int32(1))%Int32(18);ty=(j-Int32(1))÷Int32(18);gx=bx*Int32(16)+tx-Int32(1);gy=by*Int32(4)+ty-Int32(1)
        value=UInt32(0)
        if gx>=Int32(0)&&gx<W&&gy>=Int32(0)&&gy<H
            @inbounds value=input[gy*W+gx+Int32(1)]
        end
        @inbounds tile[j]=value
    end
    JACC.sync_workgroup()
    x=bx*Int32(16)+lx-Int32(1);y=by*Int32(4)+ly-Int32(1)
    if x<W&&y<H
        @inbounds output[y*W+x+Int32(1)]=median_pixel(tile,Int32(18),Int32(6),lx,ly)
    end
    nothing
end
function study_prepare(input,output,W,H)
    template=CUDA.zeros(UInt32,108);backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,4),blocks=(cld(W,16),cld(H,4)),shmem_size=432,sync=false)
    ()->JACC.parallel_for(median_tiled!,spec,(W,H),input,output,Int32(W),Int32(H),template)
end
exit(study_main(study_prepare))
