using CUDA
include(joinpath(@__DIR__,"..","extrema_math.jl"))
function extrema1_kernel!(A,R,n,order,clip)
 start=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);stride=Int32(gridDim().x*blockDim().x)
 extrema1_site!(start,stride,n,order,clip,A,R);return
end
function extrema2_kernel!(A,R,nx,ny,order,clip,axis)
 col=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);row=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
 extrema2_site!(col,row,nx,ny,order,clip,axis,A,R);return
end
function launch_extrema(model,A,R,dim,nx,ny,o,c,axis)
 if dim==1
  @cuda threads=256 blocks=cld(nx,256) extrema1_kernel!(A,R,Int32(nx),Int32(o),c)
 else
  @cuda threads=(16,16) blocks=(cld(nx,16),cld(ny,16)) extrema2_kernel!(A,R,Int32(nx),Int32(ny),Int32(o),c,Int32(axis))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:cuda_jl)
