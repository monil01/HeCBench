using CUDA,JACC
include(joinpath(@__DIR__,"..","extrema_math.jl"))
function extrema1_kernel!(i,A,R,n,order,clip,stride)
 extrema1_site!(Int32(i-1),stride,n,order,clip,A,R);nothing
end
function extrema2_kernel!(i,j,A,R,nx,ny,order,clip,axis)
 extrema2_site!(Int32(i-1),Int32(j-1),nx,ny,order,clip,axis,A,R);nothing
end
const SPECS=Dict{Tuple{Int,Int,Int},Any}()
function launch_extrema(model,A,R,dim,nx,ny,o,c,axis)
 spec=get!(SPECS,(dim,nx,ny)) do
  if dim==1
   JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=cld(nx,256),shmem_size=0,sync=false)
  else
   JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(16,16),blocks=(cld(nx,16),cld(ny,16)),shmem_size=0,sync=false)
  end
 end
 if dim==1
  stride=cld(nx,256)*256
  JACC.parallel_for(extrema1_kernel!,spec,stride,A,R,Int32(nx),Int32(o),c,Int32(stride))
 else
  JACC.parallel_for(extrema2_kernel!,spec,(cld(nx,16)*16,cld(ny,16)*16),A,R,Int32(nx),Int32(ny),Int32(o),c,Int32(axis))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:jacc)
