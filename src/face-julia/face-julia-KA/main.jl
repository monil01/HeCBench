using CUDA,KernelAbstractions
const MODEL="ka"
include("../study_device.jl")
@kernel function address_kernel!(rects,out,base,width,n)
 i=@index(Global,Linear)
 address_body!(rects,out,base,width,n,Int32(i)-Int32(1))
end
launch_address(a,b,p,w,n)=address_kernel!(CUDABackend(),256)(a,b,p,w,n;ndrange=cld(Int(n),256)*256)
include("../study_common.jl")
main()
