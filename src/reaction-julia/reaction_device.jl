@inline function pencil_position(::Val{A},lx,ly,gx,gy,M) where A
 lx=Int32(lx)-Int32(1);ly=Int32(ly)-Int32(1);gx=Int32(gx)-Int32(1);gy=Int32(gy)-Int32(1)
 if A==1
  i=lx;j=gx*Int32(2)+ly;k=gy;slot=ly*M+lx+Int32(1);coordinate=i;step=Int32(1)
 elseif A==2
  i=gx*Int32(2)+lx;j=ly;k=gy;slot=j*Int32(2)+lx+Int32(1);coordinate=j;step=Int32(2)
 else
  i=gx*Int32(2)+lx;j=gy;k=ly;slot=k*Int32(2)+lx+Int32(1);coordinate=k;step=Int32(2)
 end
 k*M*M+j*M+i+Int32(1),slot,coordinate,step
end
@inline function pencil_difference(shared,slot,coordinate,step,M)
 @inbounds center=shared[slot]
 if coordinate==0
  @inbounds answer=shared[slot+step]-center
 elseif coordinate==M-1
  @inbounds answer=shared[slot-step]-center
 else
  @inbounds answer=(shared[slot+step]-2f0*center)+shared[slot-step]
 end
 answer
end
@inline function point_site!(::Val{:lap},out,d,i,N,coef,dt,c1,c2)
 i=Int32(i)
 if i<=N;@inbounds out[i]=coef*((d.dx2[i]+d.dy2[i])+d.dz2[i]);end
 nothing
end
@inline function point_site!(::Val{:reaction},out,d,i,N,coef,dt,c1,c2)
 i=Int32(i)
 if i<=N
  @inbounds a=d.a[i];@inbounds b=d.b[i];r=(a*b)*b
  @inbounds d.ra[i]=-r+c1*(1f0-a)
  @inbounds d.rb[i]=r-(c1+c2)*b
 end
 nothing
end
@inline function point_site!(::Val{:update},out,d,i,N,coef,dt,c1,c2)
 i=Int32(i)
 if i<=N
  @inbounds d.a[i]=muladd(d.da[i]+d.ra[i],dt,d.a[i])
  @inbounds d.b[i]=muladd(d.db[i]+d.rb[i],dt,d.b[i])
 end
 nothing
end
