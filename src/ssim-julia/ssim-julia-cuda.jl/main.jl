using CUDA
include("../study_device.jl")
function ssim_kernel(A,B,O,bx,by,bz,gx,gy)
 x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
 z=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 if x<bx&&y<by&&z<bz
  @inbounds O[x+y*bx+z*bx*by+Int32(1)]=ssim_value(x,y,z,A,B,gx,gy)
 end
 nothing
end
makearray(a)=CuArray(a)
function factory(A,B,O)
 (t)->(@cuda threads=(8,8,8) blocks=(cld(t.bx,8),cld(t.by,8),cld(t.bz,8)) ssim_kernel(A,B,O,Int32(t.bx),Int32(t.by),Int32(t.bz),Int32(t.gx),Int32(t.gy)))
end
include("../study_common.jl")
ssim_study(makearray,factory)
