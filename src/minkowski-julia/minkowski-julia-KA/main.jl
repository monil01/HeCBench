using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(a,b,c,mm,nn,kk,power,invpower)
 col,row=@index(Global,NTuple)
 if col<=kk&&row<=mm
  sum=0f0
  for j in 1:nn
   @inbounds sum+=abs(a[(row-1)*nn+j]-b[(j-1)*kk+col])^power
  end
  @inbounds c[(row-1)*kk+col]=sum^invpower
 end
end
run!(a,b,c,m,n,k,p,ip)=(kernel!(get_backend(c),(16,16))(a,b,c,Int32(m),Int32(n),Int32(k),p,ip;ndrange=(k,m));nothing)
study(run!)
