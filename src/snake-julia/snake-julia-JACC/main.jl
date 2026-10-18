using CUDA,JACC,Printf
JACC.@init_backend
include("../study_device.jl")
function snake_kernel(i,A,B,O,N,T)
 if i<=N;@inbounds O[i]=snake_filter(A,B,(Int32(i)-Int32(1))*Int32(8),T);end
 nothing
end
function factory(A,B,O,N)
 spec=JACC.launch_spec(;threads=256,blocks=cld(N,256),shmem_size=0,sync=false)
 T->JACC.parallel_for(spec,cld(N,256)*256,snake_kernel,A,B,O,Int32(N),Int32(T))
end
include("../study_common.jl")
snake_study(factory)
