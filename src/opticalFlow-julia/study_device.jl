# Source software normalized mirror/bilinear image kernels and tiled Jacobi equations.
@inline function mirror(i,n)
 p=rem(i,Int32(2)*n);if p<Int32(0);p+=Int32(2)*n;end
 p>=n ? Int32(2)*n-Int32(1)-p : p
end
@inline function sample(image,w,h,stride,u,v)
 fx=muladd(u,Float32(w),-.5f0);fy=muladd(v,Float32(h),-.5f0)
 x0=unsafe_trunc(Int32,floor(fx));y0=unsafe_trunc(Int32,floor(fy));tx=fx-Float32(x0);ty=fy-Float32(y0)
 a=mirror(x0,w);b=mirror(x0+Int32(1),w);c=mirror(y0,h);d=mirror(y0+Int32(1),h)
 @inbounds begin
  v00=image[c*stride+a+Int32(1)];v10=image[c*stride+b+Int32(1)];v01=image[d*stride+a+Int32(1)];v11=image[d*stride+b+Int32(1)]
  value=(1f0-tx)*(1f0-ty)*v00
  value=muladd(tx*(1f0-ty),v10,value);value=muladd((1f0-tx)*ty,v01,value);value=muladd(tx*ty,v11,value)
 end
 value
end
@inline function body_down!(ix,iy,src,sw,sh,ss,w,h,stride,out)
 if ix>=w||iy>=h;return;end
 dx=1f0/Float32(w);dy=1f0/Float32(h);x=(Float32(ix)+.5f0)*dx;y=(Float32(iy)+.5f0)*dy
 z=sample(src,sw,sh,ss,x-dx*.25f0,y)+sample(src,sw,sh,ss,x+dx*.25f0,y)+sample(src,sw,sh,ss,x,y-dy*.25f0)+sample(src,sw,sh,ss,x,y+dy*.25f0)
 @inbounds out[ix+iy*stride+Int32(1)]=.25f0*z
 nothing
end
@inline function body_up!(ix,iy,src,sw,sh,ss,w,h,stride,scale,out)
 if ix>=w||iy>=h;return;end
 x=(Float32(ix)+.5f0)/Float32(w);y=(Float32(iy)+.5f0)/Float32(h)
 @inbounds out[ix+iy*stride+Int32(1)]=sample(src,sw,sh,ss,x,y)*scale
 nothing
end
@inline function body_warp!(ix,iy,src,w,h,stride,u,v,out)
 if ix>=w||iy>=h;return;end
 @inbounds begin
  pos=ix+iy*stride+Int32(1);x=(Float32(ix)+u[pos]+.5f0)/Float32(w);y=(Float32(iy)+v[pos]+.5f0)/Float32(h)
  out[pos]=sample(src,w,h,stride,x,y)
 end
 nothing
end
@inline function derivative(image,w,h,s,x,y,dx,dy)
 t=sample(image,w,h,s,x-2f0*dx,y-2f0*dy)
 t-=sample(image,w,h,s,x-dx,y-dy)*8f0
 t+=sample(image,w,h,s,x+dx,y+dy)*8f0
 t-=sample(image,w,h,s,x+2f0*dx,y+2f0*dy)
 t/12f0
end
@inline function body_derivatives!(ix,iy,a,b,w,h,s,gx,gy,gt)
 if ix>=w||iy>=h;return;end
 dx=1f0/Float32(w);dy=1f0/Float32(h);x=(Float32(ix)+.5f0)*dx;y=(Float32(iy)+.5f0)*dy;pos=ix+iy*s+Int32(1)
 @inbounds begin
  gx[pos]=(derivative(a,w,h,s,x,y,dx,0f0)+derivative(b,w,h,s,x,y,dx,0f0))*.5f0
  gt[pos]=sample(b,w,h,s,x,y)-sample(a,w,h,s,x,y)
  gy[pos]=(derivative(a,w,h,s,x,y,0f0,dy)+derivative(b,w,h,s,x,y,0f0,dy))*.5f0
 end
 nothing
end
@inline function body_add!(index,a,b,n,out)
 if index<n;@inbounds out[index+Int32(1)]=a[index+Int32(1)]+b[index+Int32(1)];end
 nothing
end
@inline function jacobi_load!(tx,ty,bx,by,du0,dv0,w,h,s,du,dv)
 ix=bx*Int32(32)+tx;iy=by*Int32(6)+ty;pos=min(ix,w-Int32(1))+min(iy,h-Int32(1))*s+Int32(1);sh=tx+Int32(2)+(ty+Int32(1))*Int32(34)
 @inbounds begin
  du[sh]=du0[pos];dv[sh]=dv0[pos]
  if ty==Int32(0)
   x=min(bx*Int32(32)+tx,w-Int32(1));y=max(by*Int32(6)-Int32(1),Int32(0));gm=y*s+x+Int32(1);sm=tx+Int32(2)
   du[sm]=du0[gm];dv[sm]=dv0[gm]
   y=min(by*Int32(6)+Int32(6),h-Int32(1));sm+=Int32(238);gm=y*s+x+Int32(1)
   du[sm]=du0[gm];dv[sm]=dv0[gm]
  elseif ty==Int32(1)&&tx<Int32(6)
   y=min(by*Int32(6)+tx,h-Int32(1));x=max(bx*Int32(32)-Int32(1),Int32(0));sm=Int32(35)+tx*Int32(34);gm=x+y*s+Int32(1)
   du[sm]=du0[gm];dv[sm]=dv0[gm]
   x=min(bx*Int32(32)+Int32(32),w-Int32(1));gm=y*s+x+Int32(1);sm+=Int32(33)
   du[sm]=du0[gm];dv[sm]=dv0[gm]
  end
 end
 nothing
end
@inline function jacobi_store!(tx,ty,bx,by,gx,gy,gt,w,h,s,alpha,du1,dv1,du,dv)
 ix=bx*Int32(32)+tx;iy=by*Int32(6)+ty
 if ix>=w||iy>=h;return;end
 pos=ix+iy*s+Int32(1);sh=tx+Int32(2)+(ty+Int32(1))*Int32(34)
 @inbounds begin
  su=(du[sh-Int32(1)]+du[sh+Int32(1)]+du[sh+Int32(34)]+du[sh-Int32(34)])*.25f0
  sv=(dv[sh-Int32(1)]+dv[sh+Int32(1)]+dv[sh+Int32(34)]+dv[sh-Int32(34)])*.25f0
  x=gx[pos];y=gy[pos];frac=(muladd(x,su,y*sv)+gt[pos])/(muladd(x,x,y*y)+alpha)
  du1[pos]=muladd(-x,frac,su);dv1[pos]=muladd(-y,frac,sv)
 end
 nothing
end
