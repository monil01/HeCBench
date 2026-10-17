using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(ni,nj,seed,paths,a,b,h,rth,count,err)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x;j=(blockIdx().y-1)*blockDim().y+threadIdx().y
 if i<=ni && j<=nj
  n,e=point(i,j,ni,nj,seed,paths,a,b,h,rth)
  if n!=0
   CUDA.@atomic count[1]+=n
   CUDA.@atomic err[1]+=e
  end
 end
 return
end
run!(args...)=(@cuda threads=(16,16) blocks=(cld(NI,16),cld(NJ,16)) kernel!(args...))
study(run!)
