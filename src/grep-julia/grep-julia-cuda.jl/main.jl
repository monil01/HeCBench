using CUDA,StaticArrays
include("../study_params.jl")
include("../device_nfa.jl")
allocate_model(x)=CuArray(x)
function match_kernel!(text,offset,n,regex,rt,result,match)
    shared=CuStaticSharedArray(Int32,2306)
    if threadIdx().x==1;parse_postfix!(shared,regex,rt[17]);build_nfa!(shared,match);end
    sync_threads();gid=Int((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
    match_thread!(gid,shared,text,offset,result,n)
    nothing
end
function graph_kernel!(regex,rt,match,out)
    shared=CuStaticSharedArray(Int32,2306)
    if threadIdx().x==1;parse_postfix!(shared,regex,rt[17]);build_nfa!(shared,match);end
    sync_threads()
    if threadIdx().x==1;export_graph!(shared,out,Int(blockIdx().x)-1);end
    nothing
end
launch_match!(w,regex,rt,match,result)=(@cuda threads=160 blocks=512 match_kernel!(w.text,w.offset,Int32(w.n),regex,rt,result,match))
launch_observer!(regex,rt,match,out)=(@cuda threads=160 blocks=512 graph_kernel!(regex,rt,match,out))
include("../study_common.jl")
study_main("cuda_jl")
