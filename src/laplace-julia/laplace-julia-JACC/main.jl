using CUDA,JACC
function sor_color!(row0,col0,aP,aW,aE,aS,aN,rhs,other,dest,norm,color)
 row=Int32(row0);col=Int32(col0);n=Int32(256);h=Int32(130)
 if row<=128 && col<=n
  parity=color==1 ? col&Int32(1) : (col+Int32(1))&Int32(1);opposite=Int32(1)-parity
  idx=col*h+row+Int32(1);k=Int32(2)*row-parity+n*(col-Int32(1))
  @inbounds begin
   old=dest[idx]
   res=rhs[k]+aW[k]*other[row+(col-Int32(1))*h+Int32(1)]+aE[k]*other[row+(col+Int32(1))*h+Int32(1)]+aS[k]*other[row-parity+col*h+Int32(1)]+aN[k]*other[row+opposite+col*h+Int32(1)]
   new=old*(1f0-1.85f0)+1.85f0*(res/aP[k]);dest[idx]=new;diff=new-old;norm[idx]=diff*diff
  end
 end
end
const SPEC=JACC.launch_spec(;threads=(128,2),blocks=(1,128),shmem_size=0,sync=false)
function launch_jacc(color,a,r,b,norm)
 other=color==1 ? b : r;dest=color==1 ? r : b
 JACC.parallel_for(SPEC,(128,256),sor_color!,a...,other,dest,norm,Int32(color))
end
include("../study_common.jl")
laplace_study(launch_jacc)
