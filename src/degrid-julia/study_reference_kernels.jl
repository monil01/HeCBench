using CUDA
include(joinpath(@__DIR__,"study_arithmetic.jl"))
function degrid_cuda!(out,input,img,gcf)
 tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1);base=Int32(blockIdx().x-1)*Int32(32)
 for q in ty:Int32(8):Int32(31)
  r,im=contribution(input,img,gcf,base+q,tx)
  for offset in (16,8,4,2,1)
   r+=CUDA.shfl_down_sync(UInt32(0xffffffff),r,offset);im+=CUDA.shfl_down_sync(UInt32(0xffffffff),im,offset)
  end
  if tx==0;@inbounds out[base+q+1]=Cmplx64(r,im);end
 end
 return
end
