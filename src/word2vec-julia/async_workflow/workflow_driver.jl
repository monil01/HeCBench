using Libdl
struct WFRequest
    kind::Cint;slot::Cint;ns::Cint;blocks::Cint;d::Cint;a::Cint;w::Cint;neg::Cint;t::Cint;v::Cint
    s0::Ptr{Cvoid};s1::Ptr{Cvoid};sen::Ptr{Cvoid};rng::Ptr{Cvoid};table::Ptr{Cvoid};trace::Ptr{Cvoid};counter::Ptr{Cvoid};ex::Ptr{Cvoid}
end
@assert sizeof(WFRequest)==104
const LIB=ARGS[1]
borrow(::Type{T},p,n) where T = unsafe_wrap(CuArray,CuPtr{T}(UInt(p)),(n,);own=false)
function main()
    CUDA.device!(0);CUDA.context();CUDA.stream!(CUDA.legacy_stream())
    rc=ccall((:wf_start,LIB),Cint,(Cstring,),ARGS[2]);rc==0||error("hoststart failed")
    req=Ref{WFRequest}();counts=zeros(Int,2)
    while true
        state=ccall((:wf_poll,LIB),Cint,(Ref{WFRequest},),req)
        state==-1&&break
        if state==0
            yield()
            continue
        end
        r=req[]
        dispatch!(r)
        counts[Int(r.kind)+1]+=1
        ccall((:wf_ack,LIB),Cvoid,())
    end
    rc=ccall((:wf_join,LIB),Cint,());rc==0||error("hostworkflow failed $rc")
    CUDA.device_synchronize()
    println("JULIA_WORKFLOW_DISPATCH ",counts)
end
