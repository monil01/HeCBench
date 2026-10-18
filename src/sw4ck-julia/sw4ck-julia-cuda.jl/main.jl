using CUDA
include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(bd,nk,arr)
 functions=(kernel1!,kernel2!,kernel3!,kernel4!,kernel5!)
 for k in 1:5
  head,block=stageargs(k,bd,nk);grid=(cld(Int(head[2]-head[1]),block[1]),cld(Int(head[4]-head[3]),block[2]),cld(Int(head[6]-head[5]),block[3]))
  if k==5;CUDA.@cuda threads=block blocks=grid functions[k](head...,Int32(nk),1.,-1.,arr...)
  else;CUDA.@cuda threads=block blocks=grid functions[k](head...,1.,-1.,arr...);end
 end
end
study((T,n)->CUDA.zeros(T,n),launch)
