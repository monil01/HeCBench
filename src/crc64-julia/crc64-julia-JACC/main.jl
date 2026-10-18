using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function crc_kernel!(i,sz,cs,d,words,t,it,n)
 crc_lane!(Int64(i-1),sz,cs,d,words,t,it,n,Int64(64))
 nothing
end
function launch(sz,cs,d,words,t,it,n)
 spec=JACC.launch_spec(;threads=64,blocks=1,shmem_size=0,sync=false)
 JACC.parallel_for(spec,64,crc_kernel!,sz,cs,d,words,t,it,n)
end
crc_study(launch)
