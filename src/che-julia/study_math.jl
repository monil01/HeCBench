@inline at3(x,y,z,nx,ny)=(Int(z)*Int(ny)+Int(y))*Int(nx)+Int(x)+1
@inline function lap(a,x,y,z,nx,ny,nz)
 xp=x+1==nx ? Int32(0) : x+Int32(1);xn=x==0 ? nx-Int32(1) : x-Int32(1)
 yp=y+1==ny ? Int32(0) : y+Int32(1);yn=y==0 ? ny-Int32(1) : y-Int32(1)
 zp=z+1==nz ? Int32(0) : z+Int32(1);zn=z==0 ? nz-Int32(1) : z-Int32(1)
 @inbounds center=a[at3(x,y,z,nx,ny)]
 @inbounds xx=a[at3(xp,y,z,nx,ny)]+a[at3(xn,y,z,nx,ny)]-2.0*center
 @inbounds yy=a[at3(x,yp,z,nx,ny)]+a[at3(x,yn,z,nx,ny)]-2.0*center
 @inbounds zz=a[at3(x,y,zp,nx,ny)]+a[at3(x,y,zn,nx,ny)]-2.0*center
 return xx+yy+zz
end
@inline function point!(x,y,z,::Val{stage},cold,cnew,mu,ff,nx,ny,nz) where {stage}
 if x<nx&&y<ny&&z<nz
  i=at3(x,y,z,nx,ny);ea=-2.0/9.0;eb=-2.0/9.0;eab=2.0/9.0;gamma=.5
  @inbounds if stage==1
   c=cold[i];mu[i]=4.5*((c+1.0)*ea+(c-1)*eb-2.0*c*eab)+3.0*c+c*c*c-gamma*lap(cold,x,y,z,nx,ny,nz)
  elseif stage==2
   c=cold[i];free=(9.0/4.0)*((c*c+2.0*c+1.0)*ea+(c*c-2.0*c+1.0)*eb+2.0*(1.0-c*c)*eab)+(3.0/2.0)*c*c+(3.0/12.0)*c*c*c*c
   xp=x+1==nx ? Int32(0) : x+Int32(1);xn=x==0 ? nx-Int32(1) : x-Int32(1);yp=y+1==ny ? Int32(0) : y+Int32(1);yn=y==0 ? ny-Int32(1) : y-Int32(1);zp=z+1==nz ? Int32(0) : z+Int32(1);zn=z==0 ? nz-Int32(1) : z-Int32(1)
   gx=(cold[at3(xp,y,z,nx,ny)]-cold[at3(xn,y,z,nx,ny)])/2.0;gy=(cold[at3(x,yp,z,nx,ny)]-cold[at3(x,yn,z,nx,ny)])/2.0;gz=(cold[at3(x,y,zp,nx,ny)]-cold[at3(x,y,zn,nx,ny)])/2.0
   ff[i]=free+(gamma/2.0)*(gx*gx+gy*gy+gz*gz)
  elseif stage==3
   cnew[i]=cold[i]+.01*1.0*lap(mu,x,y,z,nx,ny,nz)
  else
   tmp=cnew[i];cnew[i]=cold[i];cold[i]=tmp
  end
 end
 return
end
