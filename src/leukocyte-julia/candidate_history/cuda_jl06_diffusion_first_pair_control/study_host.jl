# Julia translation of the original detector/spline/tracker CPU equations.
# Source geometry, PI approximation, and original update/layout policies retained.
const SOURCE_PI=3.14159
const VIDEO_LIB=joinpath(@__DIR__,"study_video.so")
csin(x::Float64)=ccall((:sin,"libm.so.6"),Cdouble,(Cdouble,),x)
ccos(x::Float64)=ccall((:cos,"libm.so.6"),Cdouble,(Cdouble,),x)
csinf(x::Float32)=ccall((:sinf,"libm.so.6"),Cfloat,(Cfloat,),x)
ccosf(x::Float32)=ccall((:cosf,"libm.so.6"),Cfloat,(Cfloat,),x)
function serialsum(a)
 s=0.0;for x in a;s+=x;end;s
end
function gradients(a)
 m,n=size(a);x=zeros(m,n);y=zeros(m,n)
 for r=1:m,c=1:n
  x[r,c]=c==1 ? a[r,2]-a[r,1] : c==n ? a[r,n]-a[r,n-1] : (a[r,c+1]-a[r,c-1])/2.0
  y[r,c]=r==1 ? a[2,c]-a[1,c] : r==m ? a[m,c]-a[m-1,c] : (a[r+1,c]-a[r-1,c])/2.0
 end;x,y
end
function video_frame(video,frame,cropped,scaled,state)
 storage=fill(UInt8(165),640*480+64);data=@view storage[33:end-32]
 GC.@preserve storage begin
  count=ccall((:leuko_video_read,VIDEO_LIB),Clong,(Ptr{Cvoid},Clong,Ptr{UInt8},Clong),video,frame,pointer(data),length(data));count==307200||error("AVI frame read")
 end
 @assert all(==(165),storage[1:32])&&all(==(165),storage[end-31:end])
 record_frame!(state,frame,cropped,scaled,storage)
 top,bottom=cropped ? (110,328) : (0,479)
 a=zeros(bottom-top+1,640);scale=scaled ? 1.0/255.0 : 1.0
 for r=0:bottom-top,c=0:639;a[r+1,c+1]=Float64(data[(479-r-top)*640+c+1])*scale;end
 a
end
# Scaled-pivot LU and sequential matrix products from Meschach; no vendor BLAS.
function inverse_source(a)
 n=size(a,1);lu=copy(a);perm=collect(1:n);scale=[maximum(abs.(lu[i,:])) for i=1:n]
 for k=1:n-1
  best=0.0;im=0
  for i=k:n
   if abs(scale[i])>=0.0*abs(lu[i,k])
    v=abs(lu[i,k])/scale[i];if v>best;best=v;im=i;end
   end
  end
  if im==0;lu[k,k]=0.0;continue;end
  if im!=k
   perm[im],perm[k]=perm[k],perm[im]
   for j=1:n;lu[im,j],lu[k,j]=lu[k,j],lu[im,j];end
  end
  for i=k+1:n
   t=lu[i,k]/lu[k,k];lu[i,k]=t
   for j=k+1:n;lu[i,j]+=(-t)*lu[k,j];end
  end
 end
 out=zeros(n,n)
 for col=1:n
  x=[perm[i]==col ? 1.0 : 0.0 for i=1:n]
  # Meschach Lsolve uses sum initialized zero then inner product subtraction.
  for i=1:n
   s=0.0;for j=1:i-1;s+=lu[i,j]*x[j];end;x[i]=(x[i]-s)
  end
  for i=n:-1:1
   s=0.0;for j=i+1:n;s+=lu[i,j]*x[j];end;x[i]=(x[i]-s)/lu[i,i]
  end
  out[:,col]=x
 end;out
end
function tmatrix()
 N=9;M=4;B=zeros(N*M,N)
 for i=0:N-1,j=0:M-1
  s=Float64(j)/M
  w=((-s*s*s+3.0*s*s-3.0*s+1.0)/6.0,(3.0*s*s*s-6.0*s*s+4.0)/6.0,(-3.0*s*s*s+3.0*s*s+3.0*s+1.0)/6.0,s*s*s/6.0)
  for (k,offset) in enumerate((-1,0,1,2));B[i*M+j+1,mod(i+offset,N)+1]=w[k];end
 end
 # mtrm_mlt source is outer-product accumulation over k, ascending order.
 C=zeros(N,N)
 for i=1:N,j=1:N,k=1:N*M;C[i,j]+=B[k,i]*B[k,j];end
 inv=inverse_source(C);A=zeros(N,N*M)
 for i=1:N,j=1:N*M,k=1:N;A[i,j]+=inv[i,k]*B[j,k];end;A
end
function uniformseg(x,y)
 dist=zeros(36);perimeter=0.0
 for i=1:36
  a=mod(i,36)+1;b=mod(i-1,36)+1;dx=x[a]-x[b];dy=y[a]-y[b];dist[a]=sqrt(dx*dx+dy*dy);perimeter+=dist[a]
 end
 ds=zeros(36);ds[1]=dist[1];for i=2:36;ds[i]=ds[i-1]+dist[i];end
 xx=zeros(36);yy=zeros(36);u=perimeter/36.0
 for i=0:35
  minimum=floatmax(Float64);idx=1
  for j=1:36;t=abs(ds[j]-Float64(i)*u);if t<minimum;minimum=t;idx=j;end;end
  xx[i+1]=x[idx];yy[i+1]=y[idx]
 end;xx,yy
end
function sampling(c,derivative=false)
 N=length(c);out=zeros(N*4)
 for i=0:N-1,j=0:3
  s=Float64(j)/4.0
  if derivative
   a=c[mod(i-1,N)+1]*(-3.0*s*s+6.0*s-3.0);b=c[i+1]*(9.0*s*s-12.0*s)
   cc=c[mod(i+1,N)+1]*(-9.0*s*s+6.0*s+3.0);d=c[mod(i+2,N)+1]*(3.0*s*s)
  else
   a=c[mod(i-1,N)+1]*(-s*s*s+3.0*s*s-3.0*s+1.0);b=c[i+1]*(3.0*s*s*s-6.0*s*s+4.0)
   cc=c[mod(i+1,N)+1]*(-3.0*s*s*s+3.0*s*s+3.0*s+1.0);d=c[mod(i+2,N)+1]*s*s*s
  end
  out[i*4+j+1]=(a+b+cc+d)/6.0
 end;out
end
function interp2(a,x,y)
 m,n=size(a);out=zeros(length(x))
 for i=eachindex(x)
  @assert isfinite(x[i])&&isfinite(y[i])&&0<=x[i]<n-1&&0<=y[i]<m-1
  l=trunc(Int,x[i]);k=trunc(Int,y[i]);aa=x[i]-l;bb=y[i]-k
  out[i]=(1.0-aa)*(1.0-bb)*a[k+1,l+1]+aa*(1.0-bb)*a[k+1,l+2]+(1.0-aa)*bb*a[k+2,l+1]+aa*bb*a[k+2,l+2]
 end;out
end
function splineenergy!(cx,cy,gx,gy)
 x=sampling(cx);y=sampling(cy);xs=sampling(cx,true);ys=sampling(cy,true)
 nx=zeros(36);ny=zeros(36)
 for k=1:36;d=sqrt(xs[k]*xs[k]+ys[k]*ys[k]);nx[k]=ys[k]/d;ny[k]=-1.0*xs[k]/d;end
 x1=x+3.0*nx;y1=y+3.0*ny;x2=x-3.0*nx;y2=y+3.0*ny # actual source Y2 plus policy
 ix1=interp2(gx,x1,y1);iy1=interp2(gy,x1,y1);ix2=interp2(gx,x2,y2);iy2=interp2(gy,x2,y2)
 dx=zeros(9);dy=zeros(9)
 for i=0:8,j=0:3
  s=Float64(j)/4.0;k=i*4+j+1;D=sqrt(xs[k]*xs[k]+ys[k]*ys[k]);D3=D*D*D
  As=(-(s-1.0)*(s-1.0)*(s-1.0)/6.0,(3.0*s*s*s-6.0*s*s+4.0)/6.0,(-3.0*s*s*s+3.0*s*s+3.0*s+1.0)/6.0,s*s*s/6.0)
  Bs=((-3.0*s*s+6.0*s-3.0)/6.0,(9.0*s*s-12.0*s)/6.0,(-9.0*s*s+6.0*s+3.0)/6.0,3.0*s*s/6.0)
  xy=xs[k]*ys[k];xx=xs[k]*xs[k];yy=ys[k]*ys[k]
  for t=1:4
   A=As[t];B=Bs[t];target=mod(i+t-2,9)+1
   tx1=A-3.0*xy*B/D3;tx2=-1.0*3.0*(B/D-xx*B/D3);tx3=A+3.0*xy*B/D3;tx4=3.0*(B/D-xx*B/D3)
   ty1=3.0*(B/D-yy*B/D3);ty2=A+3.0*xy*B/D3;ty3=-1.0*3.0*(B/D-yy*B/D3);ty4=A-3.0*xy*B/D3
   dx[target]=dx[target]+tx1*ix1[k]+tx2*iy1[k]-tx3*ix2[k]-tx4*iy2[k]
   dy[target]=dy[target]+ty1*ix1[k]+ty2*iy1[k]-ty3*ix2[k]-ty4*iy2[k]
  end
 end
 # Cx,Cy physical 9x1; original loop n==1, source row1 feeds row0.
 cx[1]=cx[2]+.02*dx[1];cy[1]=cy[2]+.02*dy[1]
end
function detect_centers(gicov,dilation,gx,gy)
 coords=Tuple{Int,Int,Float64}[]
 for r=0:218,c=0:639
  score=Float64(gicov[r+1,c+1]);d=Float64(dilation[r+1,c+1])
  if abs(score)>=1e-6&&abs(d-score)<1e-6&&r>29&&r<328-110+39;push!(coords,(r,c,sqrt(score)));end
 end
 A=tmatrix();centers=Tuple{Float64,Float64}[]
 for (r,c,G) in coords
  if abs(G)>1.8
   theta=[Float64(j)*2.0*SOURCE_PI/36.0 for j=0:35]
   xx=[Float64(c)+10.0*ccos(t) for t in theta];yy=[Float64(r-40)+10.0*csin(t) for t in theta]
   x,y=uniformseg(xx,yy)
   if minimum(x)>5.0&&minimum(y)>5.0&&maximum(x)<635.0&&maximum(y)<475.0
    cx=zeros(9);cy=zeros(9)
    for i=1:9,k=1:36;cx[i]+=A[i,k]*x[k];cy[i]+=A[i,k]*y[k];end
    cy .+=40.0
    for step=1:20;splineenergy!(cx,cy,gx,gy);end
    X=sampling(cx);Y=sampling(cy.-40.0)
    # Original post-refinement quality statistic reads grad_x twice, is unused.
    ix1=interp2(gx,X,Y);iy1=interp2(gx,X,Y);xs=sampling(cx,true);ys=sampling(cy,true)
    W=zeros(36);for i=1:36;D=sqrt(xs[i]^2+ys[i]^2);W[i]=ix1[i]*(ys[i]/D)+iy1[i]*(-1.0*xs[i]/D);end
    mu=serialsum(W)/36.0;sd=sqrt(serialsum((W.-mu).^2)/36.0);unused_quality=mu/sd
    push!(centers,(serialsum(X)/36.0,serialsum(Y)/36.0+110.0))
   end
  end
 end;centers
end
function evolve(f,xc,yc,radius,angles,expected_y)
 fx,fy=gradients(f);m,n=size(f)
 for i=1:m,j=1:n
  tx=fx[i,j];ty=fy[i,j];mag=sqrt(tx*tx+ty*ty)
  fx[i,j]=mag>1e-15 ? tx/mag : 0.0;fy[i,j]=mag>1e-15 ? ty/mag : 0.0
 end
 r=copy(radius);difference=1.0;iteration=0
 while iteration<1000&&difference>.1
  oldx=xc;oldy=yc;oldr=copy(r)
  x=[xc+r[i]*ccos(angles[i]) for i=1:20];y=[yc+r[i]*csin(angles[i]) for i=1:20]
  if minimum(x)<0.0||maximum(x)>n-1||minimum(y)<0.0||maximum(y)>m-1;break;end
  length=0.0;for i=1:19;dx=x[i+1]-x[i];dy=y[i+1]-y[i];length+=sqrt(dx*dx+dy*dy);end
  dx=x[1]-x[20];dy=y[1]-y[20];length+=sqrt(dx*dx+dy*dy)
  vf=interp2(f,x,y);vfx=interp2(fx,x,y);vfy=interp2(fy,x,y)
  vfmean=serialsum(vf)/length;vfxmean=serialsum(vfx)/length;vfymean=serialsum(vfy)/length
  vfr=[(vf[i]+vfx[i]*(x[i]-xc)+vfy[i]*(y[i]-yc)-vfmean)/length for i=1:20]
  xc=xc+.2*1.0*vfxmean;yc=(yc+.2*1.0*vfymean+.2*.05*expected_y)/(1.0+.2*.05)
  rd=0.0
  for i=1:20;r[i]=(r[i]+.2*1.0*vfr[i]+.2*.2*10.0)/(1.0+.2*.2);rd+=abs(r[i]-oldr[i]);end
  difference=abs(xc-oldx)+abs(yc-oldy)+rd;iteration+=1
 end;xc,yc,r
end
function track_pipeline!(state,video,centers,frames,dispatch_mgvf)
 cells=length(centers);xc=zeros(cells,frames+1);yc=zeros(cells,frames+1);r=fill(10.0,cells,20)
 for c=1:cells;xc[c,1],yc[c,1]=centers[c];end
 angles=[(2.0*SOURCE_PI/20.0)*Float64(i) for i=0:19]
 for frame=1:frames
  state.frame=frame;image=video_frame(video,frame,false,true,state)
  patches=Matrix{Float64}[];us=Int[];vs=Int[];expected=Float64[]
  for cell=1:cells
   x=xc[cell,frame];y=yc[cell,frame];s=0.0;for i=max(frame-10,0):frame-1;s+=yc[cell,i+1];end;average=s/min(frame,10)
   u1=trunc(Int,max(x-40.0+.5,0.0));u2=trunc(Int,min(x+40.0+.5,639.0));v1=trunc(Int,max(y-20.0+1.5,0.0));v2=trunc(Int,min(y+20.0+1.5,479.0))
   sub=image[v1+1:v2+1,u1+1:u2+1];gx,gy=gradients(sub);ie=zeros(size(sub))
   for i=1:size(sub,1),j=1:size(sub,2);ie[i,j]=sqrt(gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j]);end
   lo=ie[1,1];hi=lo
   for i=1:size(ie,1),j=1:size(ie,2);v=ie[i,j];if v>hi;hi=v;elseif v<lo;lo=v;end;end
   scale=1.0/(hi-lo+2.0 ^(-52.0));for i=1:size(ie,1),j=1:size(ie,2);ie[i,j]=(ie[i,j]-lo)*scale;end
   push!(patches,ie);push!(us,u1);push!(vs,v1);push!(expected,average)
  end
  fields=dispatch_mgvf(state,patches)
  for cell=1:cells
   x,y,rr=evolve(fields[cell],xc[cell,frame]-us[cell],yc[cell,frame]-(vs[cell]-1),vec(r[cell,:]),angles,expected[cell]-(vs[cell]-1))
   xc[cell,frame+1]=x+us[cell];yc[cell,frame+1]=y+vs[cell]-1;r[cell,:]=rr
  end
  record_trajectory!(state,frame,xc[:,frame+1],yc[:,frame+1],r)
 end
end
