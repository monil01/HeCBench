using CUDA
import JACC
JACC.@init_backend
const TEMPLATE=JACC.array(zeros(Int32,128))
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=JACC.array(x)
function dot!(i,out,ref,q,n,blocks,template)
    tx=mod(Int32(i)-1,Int32(128))+1;block=(Int32(i)-1)÷Int32(128);tile=JACC.shared(template)
    @inbounds tile[tx]=dot_partial(ref,q,tx,block,blocks,n)
    JACC.sync_workgroup();stride=Int32(64)
    while stride!=0
        if tx<=stride
            @inbounds tile[tx]+=tile[tx+stride]
        end
        JACC.sync_workgroup();stride÷=2
    end
    if tx==1;@inbounds out[block+17]=tile[1];end
    return nothing
end
function cm_count!(i,scores,threshold,out,n,blocks,template)
    tx=mod(Int32(i)-1,Int32(128))+1;block=(Int32(i)-1)÷Int32(128);tile=JACC.shared(template)
    @inbounds tile[tx]=count_partial(scores,threshold,tx,block,blocks,n)
    JACC.sync_workgroup();stride=Int32(64)
    while stride!=0
        if tx<=stride
            @inbounds tile[tx]+=tile[tx+stride]
        end
        JACC.sync_workgroup();stride÷=2
    end
    if tx==1;@inbounds out[block+17]=tile[1];end
    return nothing
end
function score!(i,random,ref,scores,m,uc,set,gens)
    idx=Int32(i)-1
    random_score!(idx,random,ref,scores,m,uc,set,gens)
    return nothing
end
launch_dot!(w)=JACC.parallel_for(JACC.launch_spec(threads=128,blocks=DB,shmem_size=512,sync=false),DB*128,dot!,w.device.dot,w.device.ref,w.device.q,Int32(N),Int32(DB),TEMPLATE)
launch_scores!(w)=JACC.parallel_for(JACC.launch_spec(threads=128,blocks=SB,shmem_size=0,sync=false),SB*128,score!,w.device.random,w.device.ref,w.device.scores,Int32(M),Float32(UC),Int32(w.set),Int32(R))
launch_count!(w,threshold)=JACC.parallel_for(JACC.launch_spec(threads=128,blocks=SB,shmem_size=512,sync=false),SB*128,cm_count!,w.device.scores,threshold,w.device.count,Int32(R),Int32(SB),TEMPLATE)
include("../study_common.jl")
scratchinitial=Array(TEMPLATE);all(iszero,scratchinitial)||error("scratch initial")
study_main("jacc")
scratchfinal=Array(TEMPLATE);all(iszero,scratchfinal)||error("scratch final")
trace=joinpath(@__DIR__,"..","..","..",".porting-state","porting_logs","cm-julia-jacc","traces");mkpath(trace);scratchpath=joinpath(trace,"actual_scratch_$(getpid()).bin");write(scratchpath,vcat(scratchinitial,scratchfinal));println("STUDY_AUXILIARY scratch_initial_final256values allzero=true path=$scratchpath; every128localpositions_overwritten_by_partial_before_reduction=true")
