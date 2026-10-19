using CUDA,StaticArrays
import JACC
JACC.@init_backend
const TEMPLATE=JACC.array(zeros(Float32,512))
include("../study_params.jl")
include("../device_rng.jl")
allocate_model(x)=JACC.array(x)
function sample_kernel!(i,state,theta,y,n,a,b)
    id=i-1
    if id<N;sample_index!(id,state,theta,y,n,a,b);end
    nothing
end
function sum_kernel!(i,theta,flat,logs,template)
    tid=(i-1)%256+1;block=(i-1)÷256+1;id=(block-1)*256+tid
    scratch=JACC.shared(template)
    scratch[tid]=id<=N ? theta[16+id] : 0f0
    scratch[256+tid]=id<=N ? log(theta[16+id]) : 0f0
    JACC.sync_workgroup();stride=128
    while stride>0
        if tid<=stride;scratch[tid]+=scratch[tid+stride];scratch[256+tid]+=scratch[256+tid+stride];end
        JACC.sync_workgroup();stride÷=2
    end
    if tid==1;flat[16+block]=scratch[1];logs[16+block]=scratch[257];end
    nothing
end
launch_sample!(d,a,b)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=BLOCKS,shmem_size=0,sync=false),BLOCKS*256,sample_kernel!,d.state,d.theta,d.y,d.n,a,b)
launch_sum!(d)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=BLOCKS,shmem_size=2048,sync=false),BLOCKS*256,sum_kernel!,d.theta,d.flat,d.logs,TEMPLATE)
include("../study_common.jl")
scratchinitial=Array(TEMPLATE);all(iszero,scratchinitial)||error("scratchinitial")
study_main("jacc")
scratchfinal=Array(TEMPLATE);all(iszero,scratchfinal)||error("scratchfinal")
trace=ARGS[2];mkpath(trace);path=joinpath(trace,"actual_scratch_$(getpid()).bin");write(path,vcat(scratchinitial,scratchfinal));println("STUDY_AUXILIARY scratch_initial_final1024values allzero=true path=$path; all512positions_overwritten_beforemath=true")
