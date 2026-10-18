using CUDA,JACC
JACC.@init_backend
include("../portable_body.jl")
function pointwise_kernel(i,h,b,th,ti,bias,lg,ho,io,ci,co)
 pointwise_body!(Int32(i-1),h,b,th,ti,bias,lg,ho,io,ci,co)
 return
end
function launch(h,b,th,ti,bias,lg,ho,io,ci,co)
 specification=JACC.launch_spec(;threads=256,blocks=cld(h*b,256),shmem_size=0,sync=false)
 JACC.parallel_for(specification,h*b,pointwise_kernel,Int32(h),Int32(b),th,ti,bias,lg,ho,io,ci,co)
end
include("../study_common.jl")
run_study(launch)
