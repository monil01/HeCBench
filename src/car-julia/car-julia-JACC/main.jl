using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function car_kernel!(i,a,k,h,v,out,p)
 car_lane!(Int64(i-1),a,k,h,v,out,p,Int32(1),Int32(1),Int64(2622))
 nothing
end
function factory(a)
 p=Params(2,3,19,23,9,37,29);spec=JACC.launch_spec(;threads=256,blocks=11,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,2622,car_kernel!,a[1],a[2],a[3],a[4],a[5],p)
end
car_study(factory)
