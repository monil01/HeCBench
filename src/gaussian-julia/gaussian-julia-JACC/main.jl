using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function fan1_jacc!(index,a,m,n::Int32,t::Int32)
 id=Int32(index)-Int32(1)
 if id<n-Int32(1)-t
  row=id+t+Int32(1)
  @inbounds m[row*n+t+Int32(1)]=a[row*n+t+Int32(1)]/a[t*n+t+Int32(1)]
 end
end
function fan2_jacc!(indexcol,indexrow,a,b,m,n::Int32,t::Int32)
 col=Int32(indexcol)-Int32(1)
 row=Int32(indexrow)-Int32(1)
 if row<n-Int32(1)-t && col<n-t
  index=(row+t+Int32(1))*n+col+t+Int32(1)
  @inbounds a[index]-=m[(row+t+Int32(1))*n+t+Int32(1)]*a[t*n+col+t+Int32(1)]
  if col==Int32(0)
   @inbounds b[row+t+Int32(2)]-=m[(row+t+Int32(1))*n+t+Int32(1)]*b[t+Int32(1)]
  end
 end
end
function factory(a,b,m,n)
 first=JACC.launch_spec(;threads=256,blocks=1,shmem_size=0,sync=false)
 second=JACC.launch_spec(;threads=(16,16),blocks=(8,8),shmem_size=0,sync=false)
 return ()->begin
  for t in Int32(0):Int32(n-2)
   JACC.parallel_for(first,256,fan1_jacc!,a,m,Int32(n),t)
   JACC.parallel_for(second,(n,n),fan2_jacc!,a,b,m,Int32(n),t)
  end
 end
end
gaussian_study(factory)
