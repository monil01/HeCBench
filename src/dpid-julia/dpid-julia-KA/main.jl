using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function guidance_kernel!(input,guidance,p)
 i=UInt32(@index(Global,Linear)-1)
 guidance_lane!(i,input,guidance,p)
end
@kernel function downsampling_kernel!(input,guidance,output,p)
 i=UInt32(@index(Global,Linear)-1)
 downsample_lane!(i,input,guidance,output,p)
end
function factory(input,guidance,output,p)
 g=guidance_kernel!(CUDABackend(),128);d=downsampling_kernel!(CUDABackend(),128);n=Int(p.o_width)*Int(p.o_height)
 ()->begin
  g(input,guidance,p;ndrange=n)
  d(input,guidance,output,p;ndrange=n)
 end
end
dpid_study(factory)
