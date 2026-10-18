# STLBM source-derived D3Q19 core. Copyright University of Geneva2020; LICENSE.native.
using StaticArrays
const LBM_DIRS=((0,0,0),(-1,0,0),(0,-1,0),(0,0,-1),(-1,-1,0),(-1,1,0),(-1,0,-1),(-1,0,1),(0,-1,-1),(0,-1,1),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1))
@inline lbm_idx(x,y,z,n)=mod(x,n)+(mod(y,n)+mod(z,n)*n)*n+Int32(1)
@inline function lbm_macro(f)
 xm=f[2]+f[5]+f[6]+f[7]+f[8];xp=f[11]+f[14]+f[15]+f[16]+f[17]
 x0=f[1]+f[3]+f[4]+f[9]+f[10]+f[12]+f[13]+f[18]+f[19]
 ym=f[3]+f[5]+f[9]+f[10]+f[15];yp=f[6]+f[12]+f[14]+f[18]+f[19]
 zm=f[4]+f[7]+f[9]+f[17]+f[19];zp=f[8]+f[10]+f[13]+f[16]+f[18]
 rho=xm+xp+x0;ir=1./rho
 return rho,Float32((xp-xm)*ir),Float32((yp-ym)*ir),Float32((zp-zm)*ir)
end
@inline lbm_equ(r,t,cu,usqr)=r*t*(1+cu+cu*cu/2.-usqr)
@inline function lbm_eq!(f,rho,u0,u1,u2)
 usqr=Float64(3f0*(u0*u0+u1*u1+u2*u2))/2.;r=rho/36.
 f[1]=lbm_equ(r,12.,0.,usqr)
 f[2]=lbm_equ(r,2.,Float64(3f0*(-u0)),usqr);f[11]=f[2]-r*12*(-u0)
 f[3]=lbm_equ(r,2.,Float64(3f0*(-u1)),usqr);f[12]=f[3]-r*12*(-u1)
 f[4]=lbm_equ(r,2.,Float64(3f0*(-u2)),usqr);f[13]=f[4]-r*12*(-u2)
 cu=Float64(3f0*(-u0-u1));f[5]=lbm_equ(r,1.,cu,usqr);f[14]=f[5]-r*2*cu
 cu=Float64(3f0*(-u0+u1));f[6]=lbm_equ(r,1.,cu,usqr);f[15]=f[6]-r*2*cu
 cu=Float64(3f0*(-u0-u2));f[7]=lbm_equ(r,1.,cu,usqr);f[16]=f[7]-r*2*cu
 cu=Float64(3f0*(-u0+u2));f[8]=lbm_equ(r,1.,cu,usqr);f[17]=f[8]-r*2*cu
 cu=Float64(3f0*(-u1-u2));f[9]=lbm_equ(r,1.,cu,usqr);f[18]=f[9]-r*2*cu
 cu=Float64(3f0*(-u1+u2));f[10]=lbm_equ(r,1.,cu,usqr);f[19]=f[10]-r*2*cu
 return
end
@inline function lbm_cell!(f0,f1,vel,flags,dirs,where,x,y,z,n,omega,lid)
 if x>=n||y>=n||z>=n;return;end
 nl=n*n*n;i=lbm_idx(x,y,z,n);boundary=flags[i]
 if boundary==Int8(4)||boundary==Int8(10);return;end
 f=MVector{19,Float64}(undef);eq=MVector{19,Float64}(undef)
 for q in 0:18
  cx,cy,cz=LBM_DIRS[q+1];v=f0[q*nl+lbm_idx(x-Int32(cx),y-Int32(cy),z-Int32(cz),n)]
  if (boundary==Int8(2)||boundary==Int8(3))&&(where[i]&(Int32(1)<<q))!=0
   v=f0[((q+8)%18+1)*nl+i]
  end
  if boundary==Int8(2)
   d=dirs[i];match=(d==0&&cx==1)||(d==1&&cx== -1)||(d==2&&cy==1)||(d==3&&cy== -1)||(d==4&&cz==1)||(d==5&&cz== -1)
   if match&&q!=0
    # Exact source coefficient order 2*1./(18 or36)*3*(cx*Float32lid).
    denom=q in (1,2,3,10,11,12) ? 18. : 36.
    v+=(2*1./denom*3)*Float32(cx*Float32(lid))
   end
  end
  f[q+1]=v
 end
 rho,u0,u1,u2=lbm_macro(f);lbm_eq!(eq,rho,u0,u1,u2)
 for q in 0:18;f1[q*nl+i]=(1.-omega)*f[q+1]+omega*eq[q+1];end
 vel[i]=u0;vel[nl+i]=u1;vel[2nl+i]=u2
 return
end
