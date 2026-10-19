using CUDA,KernelAbstractions,StaticArrays
include("../study_params.jl")
include("../device_rng.jl")
allocate_model(x)=begin
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),length(x));copyto!(a,x);a
end
@kernel function sample_kernel!(state,theta,y,n,a,b)
    id=@index(Global,Linear)-1
    if id<N;sample_index!(id,state,theta,y,n,a,b);end
end
@kernel function sum_kernel!(theta,flat,logs)
    tid=@index(Local,Linear);block=@index(Group,Linear);id=(block-1)*256+tid
    scratch=@localmem Float32 (512,)
    scratch[tid]=id<=N ? theta[16+id] : 0f0
    scratch[256+tid]=id<=N ? log(theta[16+id]) : 0f0
    @synchronize()
    stride=128
    while stride>0
        if tid<=stride;scratch[tid]+=scratch[tid+stride];scratch[256+tid]+=scratch[256+tid+stride];end
        @synchronize()
        stride÷=2
    end
    if tid==1;flat[16+block]=scratch[1];logs[16+block]=scratch[257];end
end
launch_sample!(d,a,b)=sample_kernel!(CUDABackend(),256)(d.state,d.theta,d.y,d.n,a,b;ndrange=BLOCKS*256)
launch_sum!(d)=sum_kernel!(CUDABackend(),256)(d.theta,d.flat,d.logs;ndrange=BLOCKS*256)
include("../study_common.jl")
study_main("ka")
