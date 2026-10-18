using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
function launch(bd,nk,arr)
 functions=(kernel1_ka!,kernel2_ka!,kernel3_ka!,kernel4_ka!,kernel5_ka!)
 for k in 1:5
  head,block=stageargs(k,bd,nk);nd=(cld(Int(head[2]-head[1]),block[1])*block[1],cld(Int(head[4]-head[3]),block[2])*block[2],cld(Int(head[6]-head[5]),block[3])*block[3])
  if k==5;functions[k](BACKEND,block)(head...,Int32(nk),1.,-1.,arr...;ndrange=nd)
  else;functions[k](BACKEND,block)(head...,1.,-1.,arr...;ndrange=nd);end
 end
end
study((T,n)->KernelAbstractions.zeros(BACKEND,T,n),launch)
