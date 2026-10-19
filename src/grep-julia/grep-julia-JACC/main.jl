using CUDA,StaticArrays
import JACC
JACC.@init_backend
const TEMPLATE=JACC.array(zeros(Int32,2306))
include("../study_params.jl")
include("../device_nfa.jl")
allocate_model(x)=JACC.array(x)
function match_kernel!(i,text,offset,n,regex,rt,result,match,template)
    tid=(i-1)%160+1;shared=JACC.shared(template)
    if tid==1;parse_postfix!(shared,regex,rt[17]);build_nfa!(shared,match);end
    JACC.sync_workgroup();match_thread!(i-1,shared,text,offset,result,n)
    nothing
end
function graph_kernel!(i,regex,rt,match,out,template)
    tid=(i-1)%160+1;block=(i-1)÷160;shared=JACC.shared(template)
    if tid==1;parse_postfix!(shared,regex,rt[17]);build_nfa!(shared,match);end
    JACC.sync_workgroup()
    if tid==1;export_graph!(shared,out,block);end
    nothing
end
launch_match!(w,regex,rt,match,result)=JACC.parallel_for(JACC.launch_spec(threads=160,blocks=512,shmem_size=9224,sync=false),512*160,match_kernel!,w.text,w.offset,Int32(w.n),regex,rt,result,match,TEMPLATE)
launch_observer!(regex,rt,match,out)=JACC.parallel_for(JACC.launch_spec(threads=160,blocks=512,shmem_size=9224,sync=false),512*160,graph_kernel!,regex,rt,match,out,TEMPLATE)
include("../study_common.jl")
scratchinitial=Array(TEMPLATE);all(iszero,scratchinitial)||error("scratchinitial")
study_main("jacc")
scratchfinal=Array(TEMPLATE);all(iszero,scratchfinal)||error("scratchfinal")
path=joinpath(ARGS[1],"actual_scratch_$(getpid()).bin");write(path,vcat(scratchinitial,scratchfinal));println("STUDY_AUXILIARY scratch_initial_final4612values allzero=true path=$path; onlylogicallywrittenpostfix/state/metafieldsread=true")
