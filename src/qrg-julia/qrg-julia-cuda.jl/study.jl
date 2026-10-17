include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function probe!(out,inputs)
 i=threadIdx().x;if i<=14;@inbounds out[i]=moro_inv_cnd_gpu(inputs[i]);end;return
end
function launch!(kind,out,table,seed,total,n,distance,probes,inputs)
 if kind==0
  @cuda threads=256 blocks=cld(Int(total),256) qrng_kernel!(out,table,seed,total,n)
 elseif kind==1
  @cuda threads=256 blocks=cld(Int(total),256) icnd_kernel!(out,total,distance)
 else
  @cuda threads=256 blocks=1 probe!(probes,inputs)
 end
 nothing
end
qrg_study(launch!)
