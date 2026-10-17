using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function packed_atomic_jacc!(i,output,n::Int32)
 if i<=n
  bucket=mod(i-1,256)+1
  packed_half2_add!(pointer(output,bucket))
 end
 return
end
function factory(output,n)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 return ()->JACC.parallel_for(spec,n,packed_atomic_jacc!,output,Int32(n))
end
f16atomic_study(factory)
