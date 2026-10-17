using CUDA
include("../study_common.jl")
function custom_kernel(a,b,result,groups)
 lid=threadIdx().x;gid=(blockIdx().x-1)*blockDim().x+lid;stride=blockDim().x*gridDim().x;scratch=CuStaticSharedArray(eltype(a),1024)
 @inbounds scratch[lid]=thread_dot(a,b,gid,stride,groups);sync_threads()
 step=512
 while step>0
  if lid<=step;@inbounds scratch[lid]+=scratch[lid+step];end
  sync_threads();step÷=2
 end
 if lid==1;@inbounds value=scratch[1];Atomix.@atomic result[1]+=value;end
 return
end
factory(a,b,r,pad,blocks)=()->(@cuda threads=1024 blocks=blocks custom_kernel(a,b,r,pad÷4))
dp_study(factory)
