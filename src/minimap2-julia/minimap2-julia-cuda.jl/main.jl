using CUDA
include(joinpath(@__DIR__,"..","kernel_helpers.jl"))
function chain!(ret,anchors,control,maxg,parentg)
    group=CUDA.blockIdx().x; lane=CUDA.threadIdx().x
    S=CUDA.CuStaticSharedArray(UInt64,128)
    ofs=Int32(group-1); id=Int32(lane); ctl=control[Int(ofs)+1]
    avg=reinterpret(Float32,UInt32(ctl & 0xffffffff)); tile=Int32((ctl>>32)&0xffff); fresh=((ctl>>48)&0xff)!=0
    @inbounds S[id]=anchors[Int(ofs)*1089+id]
    @inbounds S[64+id]=fresh ? packed(Int32(0),Int32(-1)) : packed(maxg[Int(ofs)*64+id],parentg[Int(ofs)*64+id])
    for curridx in Int32(0):Int32(1023)
        CUDA.sync_threads()
        slot=rem(curridx,Int32(64))+Int32(1)
        @inbounds cur=S[slot]
        @inbounds v=S[64+slot]
        f=low(v); p=high(v); width=Int32((cur>>32)&0xffff)
        if width>=f
            f=width; p=Int32(-1)
        end
        CUDA.sync_threads()
        if id==slot
            @inbounds S[id]=anchors[Int(ofs)*1089+Int(curridx)+65]
            @inbounds S[64+id]=packed(Int32(0),Int32(-1))
        end
        CUDA.sync_threads()
        @inbounds nxt=S[id]
        sc=edge_score(nxt,cur,avg)
        CUDA.sync_threads()
        @inbounds v=S[64+id]
        if sc+f>=low(v)
            @inbounds S[64+id]=packed(sc+f,curridx+tile*Int32(1024))
        end
        CUDA.sync_threads()
        if id==slot
            ob=(Int(ofs)*1024+Int(curridx))*2
            @inbounds ret[ob+1]=f
            @inbounds ret[ob+2]=p
        end
    end
    CUDA.sync_threads()
    @inbounds v=S[64+id]
    @inbounds maxg[Int(ofs)*64+id]=low(v)
    @inbounds parentg[Int(ofs)*64+id]=high(v)

    nothing
end
launch!(r,a,c,m,p)=CUDA.@cuda threads=64 blocks=1792 chain!(r,a,c,m,p)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
