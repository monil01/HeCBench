using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,j,k,psi,out,sigma,template,nx,ny,nz)
 ii=Int32(i)-Int32(1);jj=Int32(j)-Int32(1);tz=mod(ii,Int32(16));ty=mod(jj,Int32(16));bx=ii÷Int32(16);by=jj÷Int32(16);bz=Int32(k)-Int32(1);sm=JACC.shared(template)
 vol=nx*ny*nz;base=Int32(20)*bx*ny*nz+Int32(14)*by*nz+Int32(14)*bz
 lastx=min(Int32(21),nx-Int32(1)-Int32(20)*bx);lasty=min(Int32(15),ny-Int32(1)-Int32(14)*by);lastz=min(Int32(15),nz-Int32(1)-Int32(14)*bz)
 if ty>lasty||tz>lastz;return;end
 prev=Int32(0);cur=Int32(1);nxt=Int32(2);slot=ty*Int32(16)+tz+Int32(1)
 @inbounds sm[cur*Int32(256)+slot]=psi[base+ty*nz+tz+1];sm[nxt*Int32(256)+slot]=psi[base+ny*nz+ty*nz+tz+1]
 dv=0.
 JACC.sync_workgroup()
 if tz>0&&tz<lastz&&ty>0&&ty<lasty
  dv-=xface(sm,cur,nxt,ty,tz,sigma,base+ny*nz+ty*nz+tz,vol)
 end
 old=prev;prev=cur;cur=nxt;nxt=old
 for xx in Int32(1):lastx-Int32(1)
  @inbounds sm[nxt*Int32(256)+slot]=psi[base+(xx+Int32(1))*ny*nz+ty*nz+tz+1]
  JACC.sync_workgroup()
  if tz>0&&tz<lastz&&ty<lasty
   yc=yface(sm,prev,cur,nxt,ty,tz,sigma,base+xx*ny*nz+(ty+Int32(1))*nz+tz,vol);dv+=yc;@inbounds sm[Int32(768)+slot]=yc
  end
  JACC.sync_workgroup()
  if tz>0&&tz<lastz&&ty>0&&ty<lasty;@inbounds dv-=sm[Int32(768)+(ty-Int32(1))*Int32(16)+tz+1];end
  JACC.sync_workgroup()
  if tz<lastz&&ty>0&&ty<lasty
   zc=zface(sm,prev,cur,nxt,ty,tz,sigma,base+xx*ny*nz+ty*nz+tz+Int32(1),vol);dv+=zc;@inbounds sm[Int32(768)+slot]=zc
  end
  JACC.sync_workgroup()
  if tz>0&&tz<lastz&&ty>0&&ty<lasty;@inbounds dv-=sm[Int32(768)+ty*Int32(16)+tz];end
  JACC.sync_workgroup()
  if tz>0&&tz<lastz&&ty>0&&ty<lasty
   xc=xface(sm,cur,nxt,ty,tz,sigma,base+(xx+Int32(1))*ny*nz+ty*nz+tz,vol);dv+=xc;@inbounds out[base+xx*ny*nz+ty*nz+tz+1]=dv;dv=-xc
  end
  JACC.sync_workgroup()
  old=prev;prev=cur;cur=nxt;nxt=old
 end
 return
end
function run!(psi,out,sigma,tmp,n)
 mod(n-Int32(2),Int32(14))==0||error("StudyrequiresfullYZtiles")
 g=(cld(n-2,20),cld(n-2,14),cld(n-2,14));JACC.parallel_for(JACC.launch_spec(threads=(16,16,1),blocks=g,shmem_size=8192,sync=false),(16g[1],16g[2],g[3]),kernel!,psi,out,sigma,tmp,n,n,n)
 return
end
study(run!)
