using CUDA,KernelAbstractions
include("../pointer_device.jl")
@kernel unsafe_indices=true function primary!(data,base,accesses,target)
 if UInt32(@index(Group,Linear)-1)==target
  p=chase_pointer(base,accesses)
  CUDA.@cuassert p != UInt64(0)
 end
end
@kernel unsafe_indices=true function diagnostic!(data,base,out,counts)
 i=Int(@index(Global,Linear))
 if i<=4;out[i]=chase_pointer(base,counts[i]);end
end
const PRIMARY=primary!(CUDABackend(),1)
const DIAGNOSTIC=diagnostic!(CUDABackend(),4)
launch(d,b,a,t,sm)=(PRIMARY(d,b,a,t;ndrange=sm);nothing)
diagnose(d,b,o,c)=(DIAGNOSTIC(d,b,o,CuArray(c);ndrange=4);nothing)
include("../study_common.jl")
pointer_study(launch,diagnose,"ka")
