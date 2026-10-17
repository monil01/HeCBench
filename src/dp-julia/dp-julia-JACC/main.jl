using JACC
include("../study_portable_common.jl")
@inline function custom_kernel(i,a,b,result,groups,stride,template)
 lid=(i-1)%1024+1;scratch=JACC.shared(template)
 @inbounds scratch[lid]=thread_dot(a,b,i,stride,groups)
 JACC.sync_workgroup()
 step=512
 while step>0
  if lid<=step;@inbounds scratch[lid]+=scratch[lid+step];end
  JACC.sync_workgroup();step÷=2
 end
 if lid==1;@inbounds value=scratch[1];Atomix.@atomic result[1]+=value;end
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(a,b,r,pad,blocks)
 template=CUDA.zeros(eltype(a),1024);spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=1024,blocks=blocks,shmem_size=1024sizeof(eltype(a)),sync=false)
 ()->JACC.parallel_for(spec,blocks*1024,custom_kernel,a,b,r,pad÷4,blocks*1024,template)
end
@inline product(i,a,b)=@inbounds a[i]*b[i]
scalar_return(a,b,r,n,pad,blocks,custom)=()->JACC.parallel_reduce(n,product,a,b;type=eltype(a))
dp_study(factory,scalar_return)
