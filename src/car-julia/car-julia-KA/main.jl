using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function car_kernel!(a,k,h,v,out,p)
 i=@index(Global,Linear)
 car_lane!(Int64(i-1),a,k,h,v,out,p,Int32(1),Int32(1),Int64(2622))
end
function factory(a)
 p=Params(2,3,19,23,9,37,29);k=car_kernel!(CUDABackend(),256)
 ()->k(a[1],a[2],a[3],a[4],a[5],p;ndrange=2622)
end
car_study(factory)
