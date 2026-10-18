using CUDA,JACC
include("../pointer_device.jl")
function primary!(i,data,base,accesses,target)
 if UInt32(i-1)==target
  p=chase_pointer(base,accesses)
  CUDA.@cuassert p != UInt64(0)
 end
 nothing
end
function diagnostic!(i,data,base,out,counts)
 if i<=4;out[i]=chase_pointer(base,counts[i]);end
 nothing
end
function launch(d,b,a,t,sm)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=1,blocks=sm,shmem_size=0,sync=false)
 JACC.parallel_for(primary!,spec,sm,d,b,a,t)
 nothing
end
const DIAG=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=4,blocks=1,shmem_size=0,sync=false)
diagnose(d,b,o,c)=(JACC.parallel_for(diagnostic!,DIAG,4,d,b,o,CuArray(c));nothing)
include("../study_common.jl")
pointer_study(launch,diagnose,"jacc")
