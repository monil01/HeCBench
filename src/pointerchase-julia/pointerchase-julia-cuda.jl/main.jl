using CUDA
include("../pointer_device.jl")
function primary!(data,base,accesses,target)
 if UInt32(blockIdx().x-1)==target
  p=chase_pointer(base,accesses)
  CUDA.@cuassert p != UInt64(0)
 end
 nothing
end
function diagnostic!(data,base,out,counts)
 i=Int(threadIdx().x)
 if i<=4;out[i]=chase_pointer(base,counts[i]);end
 nothing
end
launch(d,b,a,t,sm)=(@cuda threads=1 blocks=sm primary!(d,b,a,t);nothing)
diagnose(d,b,o,c)=(@cuda threads=4 blocks=1 diagnostic!(d,b,o,CuArray(c));nothing)
include("../study_common.jl")
pointer_study(launch,diagnose,"cuda.jl")
