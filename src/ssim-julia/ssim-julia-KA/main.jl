using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel function ssim_kernel(A,B,O,bx,by,bz,gx,gy)
 I=@index(Global,NTuple)
 x=Int32(I[1]-1);y=Int32(I[2]-1);z=Int32(I[3]-1)
 if x<bx&&y<by&&z<bz
  @inbounds O[x+y*bx+z*bx*by+Int32(1)]=ssim_value(x,y,z,A,B,gx,gy)
 end
end
function makearray(a)
 d=KernelAbstractions.allocate(CUDABackend(),Float32,(length(a),))
 copyto!(d,a)
 d
end
function factory(A,B,O)
 k=ssim_kernel(CUDABackend(),(8,8,8))
 t->k(A,B,O,Int32(t.bx),Int32(t.by),Int32(t.bz),Int32(t.gx),Int32(t.gy);ndrange=(t.bx,t.by,t.bz))
end
include("../study_common.jl")
ssim_study(makearray,factory)
