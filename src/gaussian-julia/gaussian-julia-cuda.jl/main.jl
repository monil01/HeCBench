using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function fan1_julia!(a,m,n::Int32,t::Int32)
 id=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
 if id<n-Int32(1)-t
  row=id+t+Int32(1)
  @inbounds m[row*n+t+Int32(1)]=a[row*n+t+Int32(1)]/a[t*n+t+Int32(1)]
 end
return
end
function fan2_julia!(a,b,m,n::Int32,t::Int32)
 col=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
 row=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y)-Int32(1)
 if row<n-Int32(1)-t && col<n-t
  index=(row+t+Int32(1))*n+col+t+Int32(1)
  @inbounds a[index]-=m[(row+t+Int32(1))*n+t+Int32(1)]*a[t*n+col+t+Int32(1)]
  if col==Int32(0)
   @inbounds b[row+t+Int32(2)]-=m[(row+t+Int32(1))*n+t+Int32(1)]*b[t+Int32(1)]
  end
 end
return
end
function factory(a,b,m,n)
 return ()->begin
  for t in Int32(0):Int32(n-2)
   @cuda threads=256 blocks=cld(n,256) fan1_julia!(a,m,Int32(n),t)
   @cuda threads=(16,16) blocks=(cld(n,16),cld(n,16)) fan2_julia!(a,b,m,Int32(n),t)
  end
 end
end
if abspath(PROGRAM_FILE)==@__FILE__;gaussian_study(factory);end
