using CUDA,JACC
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,j,ni,nj,seed,paths,a,b,h,rth,count,err)
 if i<=ni && j<=nj
  n,e=point(i,j,ni,nj,seed,paths,a,b,h,rth)
  if n!=0
   Atomix.@atomic count[1]+=n
   Atomix.@atomic err[1]+=e
  end
 end
end
run!(args...)=JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(2,1),shmem_size=0,sync=false),(Int(NI),Int(NJ)),kernel!,args...)
study(run!)
