using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function gemv_host!(result,data,vector,n)
 @inbounds for i=1:n
  acc=0.0
  for j=1:n;acc+=data[(i-1)*n+j]*vector[j];end
  result[i]=acc
 end
end
@inline function gemv_jacc!(index,result,data,vector,n::Int32,stride::Int32)
 i=Int32(index)
 while i<=n
  acc=0.0
  @inbounds for j=Int32(1):n;acc+=data[(j-Int32(1))*n+i]*vector[j];end
  @inbounds result[i]=acc
  i+=stride
 end
 nothing
end
# Fixed extent per immutable dictionary key; never reuse extent-cached spec across N.
const SPECS=Dict(n=>JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false) for n in STREAMUM_NS if n>=100)
function launch!(n,r,a,x)
 extent=cld(Int(n),256)*256
 JACC.parallel_for(SPECS[Int(n)],extent,gemv_jacc!,r,a,x,Int32(n),Int32(extent))
 nothing
end
streamum_study(launch!)
