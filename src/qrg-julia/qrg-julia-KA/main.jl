using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","qrg_math.jl"))
include(joinpath(@__DIR__,"..","qrg_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function qrng_ka!(out,table,seed,total,n,stride)
 idx=UInt32(@index(Global,Linear))-UInt32(1)
 qrg_body!(idx,stride,out,table,seed,total,n)
end
@kernel function icnd_ka!(out,total,distance,stride)
 idx=UInt32(@index(Global,Linear))-UInt32(1)
 icnd_body!(idx,stride,out,total,distance)
end
@kernel function probe_ka!(out,inputs)
 idx=@index(Global,Linear)
 if idx<=14;@inbounds out[idx]=moro_inv_cnd_gpu(inputs[idx]);end
end
const BACKEND=CUDABackend()
function launch!(kind,out,table,seed,total,n,distance,probes,inputs)
 extent=cld(Int(total),256)*256
 if kind==0
  qrng_ka!(BACKEND,256)(out,table,seed,total,n,UInt32(extent);ndrange=extent)
 elseif kind==1
  icnd_ka!(BACKEND,256)(out,total,distance,UInt32(extent);ndrange=extent)
 else
  probe_ka!(BACKEND,256)(probes,inputs;ndrange=256)
 end
 nothing
end
qrg_study(launch!)
