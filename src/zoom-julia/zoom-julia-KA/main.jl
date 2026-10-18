using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
function launch(a,out,pars,bc,mode,slots)
 H,W,Ho,Wo,pitch,hs,he,ws,we=pars;nd=(cld(Int(Wo),16)*16,cld(Int(Ho),16)*16,bc)
 if mode==1;zoom_in_ka!(BACKEND,(16,16,1))(a,out,pars...,Val(slots);ndrange=nd)
 else;zoom_out_ka!(BACKEND,(16,16,1))(a,out,pars...,Val(slots);ndrange=nd);zoom_edge_ka!(BACKEND,(16,16,1))(out,H,W,pitch,hs,ws,hs+Ho,ws+Wo;ndrange=(cld(Int(W),16)*16,cld(Int(H),16)*16,bc))
 end
end
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
