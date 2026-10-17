using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function median_tiled!(input,output,W,H)
    localx,localy=@index(Local,NTuple);groupx,groupy=@index(Group,NTuple);lx=Int32(localx);ly=Int32(localy);bx=Int32(groupx)-Int32(1);by=Int32(groupy)-Int32(1);lid=lx+(ly-Int32(1))*Int32(16);tile=@localmem UInt32 (108,)
    for j in lid:Int32(64):Int32(108)
        tx=(j-Int32(1))%Int32(18);ty=(j-Int32(1))÷Int32(18);gx=bx*Int32(16)+tx-Int32(1);gy=by*Int32(4)+ty-Int32(1)
        value=UInt32(0)
        if gx>=Int32(0)&&gx<W&&gy>=Int32(0)&&gy<H
            @inbounds value=input[gy*W+gx+Int32(1)]
        end
        @inbounds tile[j]=value
    end
    @synchronize
    x=bx*Int32(16)+lx-Int32(1);y=by*Int32(4)+ly-Int32(1)
    if x<W&&y<H
        @inbounds output[y*W+x+Int32(1)]=median_pixel(tile,Int32(18),Int32(6),lx,ly)
    end
    nothing
end
function study_prepare(input,output,W,H)
    k=median_tiled!(CUDABackend(),(16,4))
    ()->k(input,output,Int32(W),Int32(H);ndrange=(W,H))
end
exit(study_main(study_prepare))
