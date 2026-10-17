using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel function custom_kernel(a,b,result,groups)
 lid=@index(Local,Linear);gid=@index(Global,Linear);stride=@ndrange()[1]
 scratch=@localmem eltype(a) (1024,)
 @inbounds scratch[lid]=thread_dot(a,b,gid,stride,groups)
 @synchronize
 step=512
 while step>0
  if lid<=step;@inbounds scratch[lid]+=scratch[lid+step];end
  @synchronize
  step÷=2
 end
 if lid==1
  @inbounds value=scratch[1]
  Atomix.@atomic result[1]+=value
 end
end
function factory(a,b,r,pad,blocks)
 k=custom_kernel(CUDABackend(),1024)
 ()->k(a,b,r,pad÷4;ndrange=blocks*1024)
end
function scalar_return(a,b,r,n,pad,blocks,custom)
 ()->begin
  CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(r)),UInt8(0),sizeof(eltype(r)))
  custom();only(Array(r))
 end
end
dp_study(factory,scalar_return)
