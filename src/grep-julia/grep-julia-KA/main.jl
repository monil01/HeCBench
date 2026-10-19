using CUDA,KernelAbstractions,StaticArrays
include("../study_params.jl")
include("../device_nfa.jl")
allocate_model(x)=begin
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),length(x));copyto!(a,x);a
end
@kernel function match_kernel!(text,offset,n,regex,rt,result,match)
    tid=@index(Local,Linear);shared=@localmem Int32 (2306,)
    if tid==1;parse_postfix!(shared,regex,rt[17]);build_nfa!(shared,match);end
    @synchronize()
    gid=@index(Global,Linear)-1
    match_thread!(gid,shared,text,offset,result,n)
end
@kernel function graph_kernel!(regex,rt,match,out)
    tid=@index(Local,Linear);block=@index(Group,Linear);shared=@localmem Int32 (2306,)
    if tid==1;parse_postfix!(shared,regex,rt[17]);build_nfa!(shared,match);end
    @synchronize()
    if tid==1;export_graph!(shared,out,block-1);end
end
launch_match!(w,regex,rt,match,result)=match_kernel!(CUDABackend(),160)(w.text,w.offset,Int32(w.n),regex,rt,result,match;ndrange=512*160)
launch_observer!(regex,rt,match,out)=graph_kernel!(CUDABackend(),160)(regex,rt,match,out;ndrange=512*160)
include("../study_common.jl")
study_main("ka")
