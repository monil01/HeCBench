using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","ecl_math.jl"))
@kernel function init_ka!(n,e,idx,list,list2,pc,pc2,col,wl,sz)
 v=Int(@index(Global,Linear))-1
 ecl_init!(v,n,e,idx,list,list2,pc,pc2,col,wl,sz)
end
@kernel function large_ka!(n,idx,list2,pc,pc2,col,wl,sz)
 w=Int(@index(Global,Linear))-1
 ecl_large!(w,n,idx,list2,pc,pc2,col,wl,sz)
end
@kernel function small_ka!(n,idx,list,pc,col)
 v=Int(@index(Global,Linear))-1
 ecl_small!(v,n,idx,list,pc,col)
end
const INIT=init_ka!(CUDABackend(),512)
const LARGE=large_ka!(CUDABackend(),512)
const SMALL=small_ka!(CUDABackend(),512)
function prepare()
 dev=CUDA.device();sms=CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);m=CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);blocks=Int(sms*m÷512)
 (CuArray(IDX),CuArray(LIST),CUDA.zeros(Int32,E),CUDA.zeros(UInt32,N),CUDA.zeros(UInt32,E÷32+1),CUDA.zeros(UInt32,N),CUDA.zeros(Int32,N),CUDA.zeros(Int32,1),blocks)
end
function execute(s)
 CUDA.fill!(s[8],Int32(0));nd=s[9]*512
 INIT(N,E,s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8];ndrange=nd)
 LARGE(N,s[1],s[3],s[4],s[5],s[6],s[7],s[8];ndrange=nd)
 SMALL(N,s[1],s[2],s[4],s[6];ndrange=nd)
 -1 # Internal device convergence loops uninstrumented, not zero iterations.
end
study_main("ka",prepare,execute,s->s[6])
