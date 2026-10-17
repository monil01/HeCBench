using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","ecl_math.jl"))
const BACKEND=JACC.get_backend(:cuda)
@inline function init_jacc!(i,n,e,idx,list,list2,pc,pc2,col,wl,sz)
 ecl_init!(Int(i)-1,n,e,idx,list,list2,pc,pc2,col,wl,sz)
 return
end
@inline function large_jacc!(i,n,idx,list2,pc,pc2,col,wl,sz)
 ecl_large!(Int(i)-1,n,idx,list2,pc,pc2,col,wl,sz)
 return
end
@inline function small_jacc!(i,n,idx,list,pc,col)
 ecl_small!(Int(i)-1,n,idx,list,pc,col)
 return
end
function prepare()
 dev=CUDA.device();sms=CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);m=CUDA.attribute(dev,CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);blocks=Int(sms*m÷512)
 spec=JACC.LaunchSpec{typeof(BACKEND)}(threads=512,blocks=blocks,shmem_size=0,sync=false)
 (CuArray(IDX),CuArray(LIST),CUDA.zeros(Int32,E),CUDA.zeros(UInt32,N),CUDA.zeros(UInt32,E÷32+1),CUDA.zeros(UInt32,N),CUDA.zeros(Int32,N),CUDA.zeros(Int32,1),blocks,spec)
end
function execute(s)
 CUDA.fill!(s[8],Int32(0));nd=s[9]*512
 JACC.parallel_for(init_jacc!,s[10],nd,N,E,s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8])
 JACC.parallel_for(large_jacc!,s[10],nd,N,s[1],s[3],s[4],s[5],s[6],s[7],s[8])
 JACC.parallel_for(small_jacc!,s[10],nd,N,s[1],s[2],s[4],s[6])
 -1 # Internal device convergence counts unavailable without instrumentation.
end
study_main("jacc",prepare,execute,s->s[6])
