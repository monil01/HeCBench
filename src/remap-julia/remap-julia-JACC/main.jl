# JACC authors only source-equivalent interval scatter; shared Thrust stages remain vendor interop.
using CUDA, JACC
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_common.jl"))
function remap_device(i,second,first,out,n,k)
 if i<=k
  idx=Int(second[i])+1
  out[Int(first[idx])+1]=Int32(i-1)
  idx+=1
  while idx<=n && (i==k || idx!=Int(second[i+1])+1)
   out[Int(first[idx])+1]=Int32(i-1);idx+=1
  end
 end
 return
end
launch(s,f,o,n,k)=(JACC.parallel_for(JACC.launch_spec(;threads=256,blocks=cld(k,256),shmem_size=0,sync=false),k,remap_device,s,f,o,n,k);nothing)
study(JACC.array,launch)
