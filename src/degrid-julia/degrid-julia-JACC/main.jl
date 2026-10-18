using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
function degrid_jacc!(i,j,out,input,img,gcf,template)
 tx=Int32((i-1)%32);ty=Int32(j-1);base=Int32((i-1)÷32)*Int32(32);slot=tx+ty*Int32(32)+Int32(1);tile=JACC.shared(template)
 for cycle in 0:3
  point=base+ty+Int32(cycle)*Int32(8);r,im=contribution(input,img,gcf,point,tx);@inbounds begin;tile[slot]=r;tile[slot+256]=im;end
  JACC.sync_workgroup()
  for offset in (16,8,4,2,1)
   neighbor=tx+offset<32 ? slot+offset : slot;nr=@inbounds tile[neighbor];ni=@inbounds tile[neighbor+256]
   JACC.sync_workgroup()
   r+=nr;im+=ni;@inbounds begin;tile[slot]=r;tile[slot+256]=im;end
   JACC.sync_workgroup()
  end
  if tx==0;@inbounds out[point+1]=Cmplx64(r,im);end
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATE=JACC.zeros(Float64,512)
launch(out,input,img,gcf)=JACC.parallel_for(JACC.launch_spec(threads=(32,8),blocks=(2,1),shmem_size=4096,sync=false),(64,8),degrid_jacc!,out,input,img,gcf,TEMPLATE)
study(launch,(T,n)->JACC.array(Array{T}(undef,n)))
