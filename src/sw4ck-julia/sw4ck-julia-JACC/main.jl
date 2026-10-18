using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(bd,nk,arr)
 functions=(kernel1_jacc!,kernel2_jacc!,kernel3_jacc!,kernel4_jacc!,kernel5_jacc!)
 for k in 1:5
  head,block=stageargs(k,bd,nk);grid=(cld(Int(head[2]-head[1]),block[1]),cld(Int(head[4]-head[3]),block[2]),cld(Int(head[6]-head[5]),block[3]));nd=grid.*block;spec=JACC.launch_spec(threads=block,blocks=grid,shmem_size=0,sync=false)
  if k==5;JACC.parallel_for(spec,nd,functions[k],head...,Int32(nk),1.,-1.,arr...)
  else;JACC.parallel_for(spec,nd,functions[k],head...,1.,-1.,arr...);end
 end
end
study((T,n)->JACC.zeros(T,n),launch)
