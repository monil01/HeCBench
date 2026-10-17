using CUDA,KernelAbstractions
@kernel cpu=false function matern_ka!(n::Int32,l::Float32,sources,targets,weights,result)
 globalidx=@index(Global,NTuple);localidx=@index(Local,NTuple)
 tx=Int32(localidx[1])-Int32(1);ty=Int32(localidx[2])-Int32(1);px=Int32(globalidx[1])-Int32(1)
 if px>=n||ty>=Int32(50);return;end
 shared=@localmem Float32 (1048,)
 @inbounds begin
  if ty==0
   for i=Int32(0):Int32(2);shared[Int32(800)+tx*Int32(3)+i+Int32(1)]=targets[px*Int32(3)+i+Int32(1)];end
  end
  if tx==0
   for i=Int32(0):Int32(2);shared[Int32(848)+ty*Int32(3)+i+Int32(1)]=sources[ty*Int32(3)+i+Int32(1)];end
   shared[Int32(998)+ty+Int32(1)]=weights[ty+Int32(1)]
  end
 end
 @synchronize;squared=0f0
 @inbounds for i=Int32(0):Int32(2)
  delta=shared[Int32(800)+tx*Int32(3)+i+Int32(1)]-shared[Int32(848)+ty*Int32(3)+i+Int32(1)];squared+=delta*delta
 end
 diff=sqrt(squared)
 @inbounds shared[tx*Int32(50)+ty+Int32(1)]=(1f0+sqrt(5f0)*diff/l+5f0*squared/(3f0*l*l))*exp(-sqrt(5f0)*diff/l)*shared[Int32(998)+ty+Int32(1)]
 @synchronize
 if ty==0
  res=0f0
  @inbounds for i=Int32(0):Int32(49);res+=shared[tx*Int32(50)+i+Int32(1)];end
  @inbounds result[px+Int32(1)]=res
 end
 return
end
function launch_ka(n,l,s,t,w,out)
 matern_ka!(get_backend(out),(16,64))(n,l,s,t,w,out;ndrange=(cld(Int(n),16)*16,64))
end
include("../study_common.jl")
matern_study(launch_ka)
