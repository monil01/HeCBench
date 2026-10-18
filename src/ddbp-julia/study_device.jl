# Adapted from Rodrigo de Barros Vimieiro's 2019 GPLv3-or-later source.
# Original license and attribution are preserved in native_corrected/main.cu.
struct Geometry
 nx::Int64;ny::Int64;ns::Int64;ndx::Int64;ndy::Int64;np::Int64
 dx::Float64;dy::Float64;dz::Float64;du::Float64;dv::Float64;DSD::Float64;DDR::Float64;DAG::Float64
end
Base.@kwdef struct Aux
 np::Int64=0;nz::Int64=0;slot::Int64=0;count::Int64=0
 left::Float64=0.;step::Float64=0.;offset::Float64=0.;phi::Float64=0.;ty::Float64=0.;tz::Float64=0.
end
@inline function ddbp_lane!(I,A,p,v,::Val{1})
 x=I[1];Y=p.ndy+1;X=p.ndx+1
 if x<X;@inbounds A[1][v.np*Y*X+x*Y+1]=0.;end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{2})
 x=I[1]
 if x<v.count;@inbounds A[v.slot][x+1]=(x-v.left)*v.step+v.offset;end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{3})
 x=I[1]
 if x<p.ndy+1
  @inbounds begin
   y=A[5][x+1];z=A[6][x+1];A[12][x+1]=y*cos(v.phi)-(z-p.DDR)*sin(v.phi);A[13][x+1]=y*sin(v.phi)+(z-p.DDR)*cos(v.phi)+p.DDR
  end
 end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{4})
 x=I[1];y=I[2];Y=p.ndy+1
 if x<Y && y<p.ndx+1
  @inbounds begin
   X=A[4][y+1];dy=A[12][x+1];dz=A[13][x+1];z=A[9][v.nz+1]
   A[11][y*Y+x+1]=(X*(z-dz)-X*v.tz+X*dz)/(-v.tz+dz)
   if y==0;A[10][x+1]=((dy-v.ty)*(z-dz)-dy*v.tz+dy*dz)/(-v.tz+dz);end
  end
 end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{5})
 lane=I[1];X=p.ndx+1;Y=p.ndy+1;lines=v.slot==1 ? Y : X
 if lane<lines*p.np
  slice=div(lane,lines);line=rem(lane,lines);acc=0.
  @inbounds if v.slot==1
   for x in 0:X-1;pos=slice*X*Y+x*Y+line+1;acc+=A[1][pos];A[1][pos]=acc;end
  else
   for y in 0:Y-1;pos=slice*X*Y+line*Y+y+1;acc+=A[1][pos];A[1][pos]=acc;end
  end
 end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{6})
 px=I[1];py=I[2];Y=p.ndy+1;X=p.ndx+1;OY=p.ny+1
 if px<OY&&py<p.nx+1
  @inbounds begin
   step=A[11][(X-2)*Y+1];xn=p.ndx-A[7][py+1]/step;yn=A[8][px+1]/step-A[10][1]/step
   xd=floor(Int64,xn);yd=floor(Int64,yn);alpha=xn-xd;beta=yn-yd
   d00=0.;d10=0.;d01=0.;d11=0.
   if 0<=xn<=p.ndx&&0<=yn<=p.ndy;d00=A[1][v.np*X*Y+xd*Y+yd+1];end
   if 0<xd+1<=p.ndx&&0<=yn<=p.ndy;d10=A[1][v.np*X*Y+(xd+1)*Y+yd+1];end
   if 0<=xn<=p.ndx&&0<yd+1<=p.ndy;d01=A[1][v.np*X*Y+xd*Y+yd+2];end
   if 0<xd+1<=p.ndx&&0<yd+1<=p.ndy;d11=A[1][v.np*X*Y+(xd+1)*Y+yd+2];end
   t1=alpha*d10+(-d00*alpha+d00);t2=alpha*d11+(-d01*alpha+d01);A[2][py*OY+px+1]=beta*t2+(-t1*beta+t1)
  end
 end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{7})
 px=I[1];py=I[2];OY=p.ny+1
 if px<p.ny&&py<p.nx
  @inbounds begin
   gamma=atan((A[7][py+1]+p.dx/2)/(v.tz-A[9][v.nz+1]));alpha=atan((A[8][px+1]+p.dy/2-v.ty)/(v.tz-A[9][v.nz+1]))
   a=A[2][py*OY+px+1];b=A[2][(py+1)*OY+px+1];c=A[2][py*OY+px+2];d=A[2][(py+1)*OY+px+2]
   if c==0&&d==0;c=a;d=b;end
   A[3][v.nz*p.nx*p.ny+py*p.ny+px+1]+=(d-c-b+a)*(p.du*p.dv*p.dz/(cos(alpha)*cos(gamma)*p.dx*p.dy))
  end
 end
 nothing
end
@inline function ddbp_lane!(I,A,p,v,::Val{8})
 x,y,z=I
 if x<p.ny&&y<p.nx&&z<p.ns;@inbounds A[3][z*p.nx*p.ny+y*p.ny+x+1]/=p.np;end
 nothing
end
