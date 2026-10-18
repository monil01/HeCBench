using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","extrema_math.jl"))
@kernel unsafe_indices=true function extrema1_kernel!(A,R,n,order,clip,stride)
 start=Int32(@index(Global,Linear)-1)
 extrema1_site!(start,stride,n,order,clip,A,R)
end
@kernel unsafe_indices=true function extrema2_kernel!(A,R,nx,ny,order,clip,axis)
 g=@index(Global,NTuple)
 extrema2_site!(Int32(g[1]-1),Int32(g[2]-1),nx,ny,order,clip,axis,A,R)
end
function launch_extrema(model,A,R,dim,nx,ny,o,c,axis)
 if dim==1
  ndrange=cld(nx,256)*256
  extrema1_kernel!(CUDABackend(),256)(A,R,Int32(nx),Int32(o),c,Int32(ndrange);ndrange=ndrange)
 else
  extrema2_kernel!(CUDABackend(),(16,16))(A,R,Int32(nx),Int32(ny),Int32(o),c,Int32(axis);ndrange=(cld(nx,16)*16,cld(ny,16)*16))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:ka)
