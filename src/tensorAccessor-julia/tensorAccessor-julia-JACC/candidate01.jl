using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const SPEC=JACC.LaunchSpec{typeof(JACC.backend)}(threads=256,shmem_size=0,sync=false)
@inline function raw_study!(i,nr,nc,r,m,v)
 i=Int64(i)
 if i<=nr
  val=0f0
  @inbounds for j in Int64(1):nc;val=muladd(m[(i-1)*nc+j],v[j],val);end
  @inbounds r[i]=val
 end
 return
end
@inline function packed_study!(i,r,m,v)
 i=Int64(i)
 if i<=size(r,1)
  val=0f0
  for j in Int64(1):size(m,2);val=muladd(m[i][j],v[j],val);end
  r[i]=val
 end
 return
end
function launch!(mode,f,d)
 if mode==1
  JACC.parallel_for(raw_study!,SPEC,f[1],Int64(f[1]),Int64(f[2]),d[3],d[1],d[2])
 else
  JACC.parallel_for(packed_study!,SPEC,f[1],d[5],d[6],d[7])
 end
end
study_main(launch!)
