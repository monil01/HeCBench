using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function guidance_kernel!(i,input,guidance,p)
 guidance_lane!(UInt32(i-1),input,guidance,p)
 nothing
end
function downsampling_kernel!(i,input,guidance,output,p)
 downsample_lane!(UInt32(i-1),input,guidance,output,p)
 nothing
end
function factory(input,guidance,output,p)
 n=Int(p.o_width)*Int(p.o_height);spec=JACC.launch_spec(;threads=128,blocks=cld(n,128),shmem_size=0,sync=false)
 ()->begin
  JACC.parallel_for(spec,n,guidance_kernel!,input,guidance,p)
  JACC.parallel_for(spec,n,downsampling_kernel!,input,guidance,output,p)
 end
end
dpid_study(factory)
