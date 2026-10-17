using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function fan1_ka!(a,m,n::Int32,t::Int32)
 id=Int32(@index(Global,Linear))-Int32(1)
 if id<n-Int32(1)-t
  row=id+t+Int32(1)
  @inbounds m[row*n+t+Int32(1)]=a[row*n+t+Int32(1)]/a[t*n+t+Int32(1)]
 end
end
@kernel function fan2_ka!(a,b,m,n::Int32,t::Int32)
 col=Int32(@index(Global,NTuple)[1])-Int32(1)
 row=Int32(@index(Global,NTuple)[2])-Int32(1)
 if row<n-Int32(1)-t && col<n-t
  index=(row+t+Int32(1))*n+col+t+Int32(1)
  @inbounds a[index]-=m[(row+t+Int32(1))*n+t+Int32(1)]*a[t*n+col+t+Int32(1)]
  if col==Int32(0)
   @inbounds b[row+t+Int32(2)]-=m[(row+t+Int32(1))*n+t+Int32(1)]*b[t+Int32(1)]
  end
 end
end
function factory(a,b,m,n)
 first=fan1_ka!(get_backend(a),256);second=fan2_ka!(get_backend(a),(16,16))
 return ()->begin
  for t in Int32(0):Int32(n-2)
   first(a,m,Int32(n),t;ndrange=256)
   second(a,b,m,Int32(n),t;ndrange=(n,n))
  end
 end
end
gaussian_study(factory)
