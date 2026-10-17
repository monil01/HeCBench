using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","dct8_math.jl"))
@kernel function study_tile!(dst,src,s,h,w,dir)
 l=@index(Local,NTuple);g=@index(Group,NTuple)
 tile=@localmem Float32 (528,)
 lx=l[1]-1;ly=8*(l[2]-1);mx=lx%8;gx=(g[1]-1)*32+lx;gy=(g[2]-1)*16+ly
 lv=ly*33+lx+1;lh=(ly+mx)*33+lx-mx+1;valid=gx-mx+7<w&&gy+7<h
 if valid
  for i in 0:7;@inbounds tile[lv+i*33]=src[(gy+i)*s+gx+1];end
 end
 @synchronize
 if valid
  d=ntuple(i->(@inbounds tile[lh+i-1]),Val(8));z=study_butterfly(d,dir)
  for i in 0:7;@inbounds tile[lh+i]=z[i+1];end
 end
 @synchronize
 if valid
  d=ntuple(i->(@inbounds tile[lv+(i-1)*33]),Val(8));z=study_butterfly(d,dir)
  for i in 0:7;@inbounds dst[(gy+i)*s+gx+1]=z[i+1];end
 end
end
function study_launch!(y,x,s,h,w,dir)
 backend=KernelAbstractions.get_backend(y)
 study_tile!(backend,(32,2))(y,x,s,h,w,dir;ndrange=(cld(w,32)*32,cld(h,16)*2))
end
study_main(study_launch!)
