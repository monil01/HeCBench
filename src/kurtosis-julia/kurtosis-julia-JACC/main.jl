using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
const KTEMPLATE=CUDA.fill(zero(KResult),256)
function kreduce_jacc!(index,input,out,n::Int32,mode,template)
 i=Int32(index);tid=(i-Int32(1))%Int32(256)+Int32(1);block=(i-Int32(1))÷Int32(256)+Int32(1);tile=JACC.shared(template)
 @inbounds tile[tid]=i<=n ? kload(input,i,mode) : zero(KResult)
 JACC.sync_workgroup()
 stride=Int32(128)
 while stride>0
  if tid<=stride;@inbounds tile[tid]=kmerge(tile[tid],tile[tid+stride]);end
  JACC.sync_workgroup();stride>>=1
 end
 if tid==1;@inbounds out[block]=tile[1];end
 return
end
function launch!(x,partial,out,n,B)
 s=JACC.launch_spec(;threads=256,blocks=B,shmem_size=5120,sync=false)
 JACC.parallel_for(s,B*256,kreduce_jacc!,x,partial,Int32(n),Val(0),KTEMPLATE)
 s2=JACC.launch_spec(;threads=256,blocks=1,shmem_size=5120,sync=false)
 JACC.parallel_for(s2,256,kreduce_jacc!,partial,out,Int32(B),Val(1),KTEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
kurtosis_study(launch!)
