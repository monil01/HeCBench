using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(psi,out,sigma,nx,ny,nz)
 tz=threadIdx().x-Int32(1);ty=threadIdx().y-Int32(1);bx=blockIdx().x-Int32(1);by=blockIdx().y-Int32(1);bz=blockIdx().z-Int32(1);sm=CuStaticSharedArray(Float64,1024)
 vol=nx*ny*nz;base=Int32(20)*bx*ny*nz+Int32(14)*by*nz+Int32(14)*bz
 lastx=min(Int32(21),nx-Int32(1)-Int32(20)*bx);lasty=min(Int32(15),ny-Int32(1)-Int32(14)*by);lastz=min(Int32(15),nz-Int32(1)-Int32(14)*bz)
 if ty>lasty||tz>lastz;return;end
 prev=Int32(0);cur=Int32(1);nxt=Int32(2);slot=ty*Int32(16)+tz+Int32(1)
 @inbounds sm[cur*Int32(256)+slot]=psi[base+ty*nz+tz+1];sm[nxt*Int32(256)+slot]=psi[base+ny*nz+ty*nz+tz+1]
 dv=0.
 sync_threads()
 if tz>0&&tz<lastz&&ty>0&&ty<lasty
  dv-=xface(sm,cur,nxt,ty,tz,sigma,base+ny*nz+ty*nz+tz,vol)
 end
 old=prev;prev=cur;cur=nxt;nxt=old
 for xx in Int32(1):lastx-Int32(1)
  @inbounds sm[nxt*Int32(256)+slot]=psi[base+(xx+Int32(1))*ny*nz+ty*nz+tz+1]
  sync_threads()
  if tz>0&&tz<lastz&&ty<lasty
   yc=yface(sm,prev,cur,nxt,ty,tz,sigma,base+xx*ny*nz+(ty+Int32(1))*nz+tz,vol);dv+=yc;@inbounds sm[Int32(768)+slot]=yc
  end
  sync_threads()
  if tz>0&&tz<lastz&&ty>0&&ty<lasty;@inbounds dv-=sm[Int32(768)+(ty-Int32(1))*Int32(16)+tz+1];end
  sync_threads()
  if tz<lastz&&ty>0&&ty<lasty
   zc=zface(sm,prev,cur,nxt,ty,tz,sigma,base+xx*ny*nz+ty*nz+tz+Int32(1),vol);dv+=zc;@inbounds sm[Int32(768)+slot]=zc
  end
  sync_threads()
  if tz>0&&tz<lastz&&ty>0&&ty<lasty;@inbounds dv-=sm[Int32(768)+ty*Int32(16)+tz];end
  sync_threads()
  if tz>0&&tz<lastz&&ty>0&&ty<lasty
   xc=xface(sm,cur,nxt,ty,tz,sigma,base+(xx+Int32(1))*ny*nz+ty*nz+tz,vol);dv+=xc;@inbounds out[base+xx*ny*nz+ty*nz+tz+1]=dv;dv=-xc
  end
  sync_threads()
  old=prev;prev=cur;cur=nxt;nxt=old
 end
 return
end
run!(psi,out,sigma,tmp,n)=(@cuda threads=(16,16,1) blocks=(cld(n-2,20),cld(n-2,14),cld(n-2,14)) kernel!(psi,out,sigma,n,n,n);nothing)
study(run!)
