using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function raw_study!(nr,nc,r,m,v)
 i=Int64(@index(Global,Linear))
 if i<=nr
  val=0f0
  @inbounds for j in Int64(1):nc;val=muladd(m[(i-1)*nc+j],v[j],val);end
  @inbounds r[i]=val
 end
end
@kernel function packed_study!(r,m,v)
 i=Int64(@index(Global,Linear))
 if i<=size(r,1)
  val=0f0
  for j in Int64(1):size(m,2);val=muladd(m[i][j],v[j],val);end
  r[i]=val
 end
end
const RAW_KERNEL=raw_study!(CUDABackend(),256)
const PACKED_KERNEL=packed_study!(CUDABackend(),256)
function launch!(mode,f,d)
 if mode==1
  RAW_KERNEL(Int64(f[1]),Int64(f[2]),d[3],d[1],d[2];ndrange=f[1])
 else
  PACKED_KERNEL(d[5],d[6],d[7];ndrange=f[1])
 end
end
study_main(launch!)
