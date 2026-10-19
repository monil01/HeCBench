using CUDA,StaticArrays
include("../study_params.jl")
include("../device_rng.jl")
allocate_model(x)=CuArray(x)
function sample_kernel!(state,theta,y,n,a,b)
    id=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    if id<N;sample_index!(id,state,theta,y,n,a,b);end
    nothing
end
function sum_kernel!(theta,flat,logs)
    tid=Int(threadIdx().x);block=Int(blockIdx().x);id=(block-1)*256+tid
    scratch=CuStaticSharedArray(Float32,512)
    scratch[tid]=id<=N ? theta[16+id] : 0f0
    scratch[256+tid]=id<=N ? log(theta[16+id]) : 0f0
    sync_threads();stride=128
    while stride>0
        if tid<=stride;scratch[tid]+=scratch[tid+stride];scratch[256+tid]+=scratch[256+tid+stride];end
        sync_threads();stride÷=2
    end
    if tid==1;flat[16+block]=scratch[1];logs[16+block]=scratch[257];end
    nothing
end
launch_sample!(d,a,b)=(@cuda threads=256 blocks=BLOCKS sample_kernel!(d.state,d.theta,d.y,d.n,a,b))
launch_sum!(d)=(@cuda threads=256 blocks=BLOCKS sum_kernel!(d.theta,d.flat,d.logs))
include("../study_common.jl")
study_main("cuda_jl")
