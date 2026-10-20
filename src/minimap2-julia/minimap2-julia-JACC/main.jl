using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","kernel_helpers.jl"))
function chain!(index,ret,anchors,control,maxg,parentg,template)
    group=div(Int(index)-1,64)+1; lane=rem(Int(index)-1,64)+1
    S=JACC.shared(template)
    ofs=Int32(group-1); id=Int32(lane); ctl=control[Int(ofs)+1]
    avg=reinterpret(Float32,UInt32(ctl & 0xffffffff)); tile=Int32((ctl>>32)&0xffff); fresh=((ctl>>48)&0xff)!=0
    @inbounds S[id]=anchors[Int(ofs)*1089+id]
    @inbounds S[64+id]=fresh ? packed(Int32(0),Int32(-1)) : packed(maxg[Int(ofs)*64+id],parentg[Int(ofs)*64+id])
    for curridx in Int32(0):Int32(1023)
        JACC.sync_workgroup()
        slot=rem(curridx,Int32(64))+Int32(1)
        @inbounds cur=S[slot]
        @inbounds v=S[64+slot]
        f=low(v); p=high(v); width=Int32((cur>>32)&0xffff)
        if width>=f
            f=width; p=Int32(-1)
        end
        JACC.sync_workgroup()
        if id==slot
            @inbounds S[id]=anchors[Int(ofs)*1089+Int(curridx)+65]
            @inbounds S[64+id]=packed(Int32(0),Int32(-1))
        end
        JACC.sync_workgroup()
        @inbounds nxt=S[id]
        sc=edge_score(nxt,cur,avg)
        JACC.sync_workgroup()
        @inbounds v=S[64+id]
        if sc+f>=low(v)
            @inbounds S[64+id]=packed(sc+f,curridx+tile*Int32(1024))
        end
        JACC.sync_workgroup()
        if id==slot
            ob=(Int(ofs)*1024+Int(curridx))*2
            @inbounds ret[ob+1]=f
            @inbounds ret[ob+2]=p
        end
    end
    JACC.sync_workgroup()
    @inbounds v=S[64+id]
    @inbounds maxg[Int(ofs)*64+id]=low(v)
    @inbounds parentg[Int(ofs)*64+id]=high(v)

    nothing
end
const TEMPLATE=JACC.array(zeros(UInt64,128))
all(iszero,Array(TEMPLATE))||error("GPUtemplateidentity")
println("STUDY_BACKEND_INPUT actual_GPU_zero_UInt64_shared_template128_checked=true; allsharedpositions_overwritten_before_algorithmic_use=true")
launch!(r,a,c,m,p)=JACC.parallel_for(JACC.launch_spec(threads=64,blocks=1792,shmem_size=1024,sync=false),1792*64,chain!,r,a,c,m,p,TEMPLATE)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
