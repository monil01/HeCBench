using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
@kernel function degrid_ka!(out,input,img,gcf)
 lx,ly=@index(Local,NTuple);bx,by=@index(Group,NTuple);tx=Int32(lx-1);ty=Int32(ly-1);base=Int32(bx-1)*Int32(32);slot=Int32(lx)+(Int32(ly)-Int32(1))*Int32(32);tile=@localmem Float64 (512,)
 for cycle in 0:3
  point=base+ty+Int32(cycle)*Int32(8);r,im=contribution(input,img,gcf,point,tx);@inbounds begin;tile[slot]=r;tile[slot+256]=im;end
  @synchronize
  for offset in (16,8,4,2,1)
   neighbor=tx+offset<32 ? slot+offset : slot;nr=@inbounds tile[neighbor];ni=@inbounds tile[neighbor+256]
   @synchronize
   r+=nr;im+=ni;@inbounds begin;tile[slot]=r;tile[slot+256]=im;end
   @synchronize
  end
  if tx==0;@inbounds out[point+1]=Cmplx64(r,im);end
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const DEGRID=degrid_ka!(BACKEND,(32,8))
launch(out,input,img,gcf)=DEGRID(out,input,img,gcf;ndrange=(64,8))
study(launch,(T,n)->KernelAbstractions.allocate(BACKEND,T,n))
