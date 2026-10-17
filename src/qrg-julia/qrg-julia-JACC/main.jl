using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","qrg_math.jl"))
include(joinpath(@__DIR__,"..","qrg_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function qrng_jacc!(i,out,table,seed,total,n,stride)
 qrg_body!(UInt32(i)-UInt32(1),stride,out,table,seed,total,n)
end
@inline function icnd_jacc!(i,out,total,distance,stride)
 icnd_body!(UInt32(i)-UInt32(1),stride,out,total,distance)
end
@inline function probe_jacc!(i,out,inputs)
 if i<=14;@inbounds out[i]=moro_inv_cnd_gpu(inputs[i]);end
 nothing
end
const PRIMARY_SPEC=JACC.launch_spec(threads=256,blocks=769,shmem_size=0,sync=false)
const PROBE_SPEC=JACC.launch_spec(threads=256,blocks=1,shmem_size=0,sync=false)
function launch!(kind,out,table,seed,total,n,distance,probes,inputs)
 extent=769*256
 if kind==0
  JACC.parallel_for(PRIMARY_SPEC,extent,qrng_jacc!,out,table,seed,total,n,UInt32(extent))
 elseif kind==1
  JACC.parallel_for(PRIMARY_SPEC,extent,icnd_jacc!,out,total,distance,UInt32(extent))
 else
  JACC.parallel_for(PROBE_SPEC,256,probe_jacc!,probes,inputs)
 end
 nothing
end
qrg_study(launch!)
