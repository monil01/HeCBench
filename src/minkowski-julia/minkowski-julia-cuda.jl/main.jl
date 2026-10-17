using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(a,b,c,mm,nn,kk,power,invpower)
 col=(blockIdx().x-1)*blockDim().x+threadIdx().x
 row=(blockIdx().y-1)*blockDim().y+threadIdx().y
 if col<=kk&&row<=mm
  sum=0f0
  for j in 1:nn
   @inbounds sum+=abs(a[(row-1)*nn+j]-b[(j-1)*kk+col])^power
  end
  @inbounds c[(row-1)*kk+col]=sum^invpower
 end
 return
end
run!(a,b,c,m,n,k,p,ip)=(@cuda threads=(16,16) blocks=(cld(k,16),cld(m,16)) kernel!(a,b,c,Int32(m),Int32(n),Int32(k),p,ip);nothing)
study(run!)
