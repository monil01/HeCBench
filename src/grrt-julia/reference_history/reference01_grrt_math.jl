#=
/*
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License
  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 
 */
=#
# Complete Odyssey function translation; source constants/ISCO Float32 boundary retained.
using StaticArrays
const GR_PI=3.14159265
const GR_Cc=2.99792458e10
const GR_MsunBH=4.3e6
const GR_rgeo=1.4774e5
const GR_distance=8500.0*3.086e18
const GR_h=6.6260755e-27
const GR_kB=1.380658e-16
const GR_e=4.8032068e-10
const GR_me=9.1093897e-28
const GR_mp=1.6726231e-24
const GR_K2=(-10.747001,-9.362569,-8.141373,-7.061568,-6.104060,-5.252153,-4.491244,-3.808555,-3.192909,-2.634534,-2.124893,-1.656543,-1.223007,-.818668,-.438676,-.078863,.264332,.593930,.912476,1.222098,1.524560,1.821311,2.113537,2.402193,2.688050,2.971721,3.253692,3.534347,3.813984,4.092839,4.371092,4.648884,4.926323,5.203493,5.480457,5.757264,6.033952,6.310550,6.587078,6.863554,7.139990,7.416395,7.692778,7.969143,8.245495,8.521837,8.798171,9.074500,9.350824,9.627144)
@inline function gr_geodesic!(V,vin,y,d)
 @inbounds begin
 r=y[1];theta=y[2];pr=y[5];ptheta=y[6];a=vin[1];a2=V[3];L=V[6];kappa=V[7]
 r2=r*r;twor=2.0*r;st=sin(theta);ct=cos(theta);cos2=ct*ct;sin2=st*st;sigma=r2+a2*cos2;delta=r2-twor+a2;sd=sigma*delta;siginv=1.0/sigma;bot=1.0/sd
 if st<1e-8;st=1e-8;sin2=1e-16;end
 d[1]=-pr*delta*siginv;d[2]=-ptheta*siginv;d[3]=-(twor*a+(sigma-twor)*L/sin2)*bot;d[4]=-(1.0+(twor*(r2+a2)-twor*a*L)*bot)
 d[5]=-(((r-1.0)*(-kappa)+twor*(r2+a2)-2.0*a*L)*bot-2.0*pr*pr*(r-1.0)*siginv)
 d[6]=-st*ct*(L*L/(sin2*sin2)-a2)*siginv
 end
 nothing
end
function gr_rkstep!(V,vin,y,d,h,out,err)
 ak=MVector{6,Float64}(undef);t1=MVector{6,Float64}(undef);t2=MVector{6,Float64}(undef);t3=MVector{6,Float64}(undef);t4=MVector{6,Float64}(undef);t5=MVector{6,Float64}(undef)
 @inbounds begin
 for i in 1:6
  hd=h*d[i];yi=y[i];t1[i]=yi+.2*hd;t2[i]=yi+(3.0/40.0)*hd;t3[i]=yi+.3*hd;t4[i]=yi-(11.0/54.0)*hd;t5[i]=yi+(1631.0/55296.0)*hd;out[i]=yi+(37.0/378.0)*hd;err[i]=((37.0/378.0)-(2825.0/27648.0))*hd
 end
 gr_geodesic!(V,vin,t1,ak)
 for i in 1:6
  yt=h*ak[i];t2[i]+=(9.0/40.0)*yt;t3[i]-=.9*yt;t4[i]+=2.5*yt;t5[i]+=(175.0/512.0)*yt
 end
 gr_geodesic!(V,vin,t2,ak)
 for i in 1:6
  yt=h*ak[i];t3[i]+=1.2*yt;t4[i]-=(70.0/27.0)*yt;t5[i]+=(575.0/13824.0)*yt;out[i]+=(250.0/621.0)*yt;err[i]+=((250.0/621.0)-(18575.0/48384.0))*yt
 end
 gr_geodesic!(V,vin,t3,ak)
 for i in 1:6
  yt=h*ak[i];t4[i]+=(35.0/27.0)*yt;t5[i]+=(44275.0/110592.0)*yt;out[i]+=(125.0/594.0)*yt;err[i]+=((125.0/594.0)-(13525.0/55296.0))*yt
 end
 gr_geodesic!(V,vin,t4,ak)
 for i in 1:6
  yt=h*ak[i];t5[i]+=(253.0/4096.0)*yt;err[i]-=(277.0/14336.0)*yt
 end
 gr_geodesic!(V,vin,t5,ak)
 for i in 1:6
  yt=h*ak[i];out[i]+=(512.0/1771.0)*yt;err[i]+=((512.0/1771.0)-.25)*yt
 end
 end
 nothing
end
function gr_rk5!(V,vin,y,d,htry,escal,yscal)
 err=MVector{6,Float64}(undef);tmp=MVector{6,Float64}(undef);h=htry;errmax=0.0
 @inbounds begin
 while true
  gr_rkstep!(V,vin,y,d,h,tmp,err);errmax=0.0
  for i in 1:6
   val=abs(err[i]/yscal[i]);if val>errmax;errmax=val;end
  end
  errmax*=escal;if errmax<=1.0;break;end
  htemp=.9*h/sqrt(sqrt(errmax));h*=.1
  if h>=0
   if htemp>h;h=htemp;end
  else
   if htemp<h;h=htemp;end
  end
 end
 hnext=errmax>1.89e-4 ? .9*h*errmax^(-.2) : 5.0*h
 for i in 1:6;y[i]=tmp[i];end
 end
 hnext,h
end
function gr_initial!(V,vin,y0,ydot)
 @inbounds begin
 a=vin[1];r0=V[1];theta0=V[2];a2=V[3];alpha=V[8];beta=V[9]
 x=sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);yy=alpha;z=r0*cos(theta0)+beta*sin(theta0);w=x*x+yy*yy+z*z-a2
 y0[1]=sqrt((w+sqrt(w*w+(2.0*a*z)*(2.0*a*z)))/2.0);y0[2]=acos(z/y0[1]);y0[3]=atan(yy,x);y0[4]=0
 r=y0[1];theta=y0[2];phi=y0[3];sigma=r*r+(a*cos(theta))*(a*cos(theta));u=sqrt(a2+r*r);v=-sin(theta0)*cos(phi);zdot=-1.0
 rdot=zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;tdot=zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;pdot=zdot*sin(theta0)*sin(phi)/(u*sin(theta))
 ydot[1]=rdot;ydot[2]=tdot;ydot[3]=pdot;st=sin(theta);sin2=st*st;r2=r*r;delta=r2-2.0*r+a2;s1=sigma-2.0*r
 y0[5]=rdot*sigma/delta;y0[6]=tdot*sigma;energy2=s1*(rdot*rdot/delta+tdot*tdot)+delta*sin2*pdot*pdot;energy=sqrt(energy2)
 y0[5]/=energy;y0[6]/=energy;L=((sigma*delta*pdot-2.0*a*r*energy)*sin2/s1)/energy;V[6]=L;V[7]=y0[6]*y0[6]+a2*sin2+L*L/sin2
 end
 nothing
end
@inline function gr_ISCO(vin)
 a=vin[1];z1=1+(1-a*a)^(1/3.0)*(1+a)^(1/3.0)+(1-a)^(1/3.0);z2=sqrt(3*a*a+z1*z1)
 Float32(3.0+z2-sqrt((3-z1)*(3+z1+2*z2)))
end
@inline function gr_K2(te)
 if te>85.0;return 2.0*te*te;end
 if te<.1;return exp(-11.0);end
 d=50.0*(log(te/.1)/log(100.0/.1));i=floor(Int,d)
 @inbounds tab=(1.0-(d-i))*GR_K2[i+1]+(d-i)*GR_K2[i+2]
 exp(tab)
end
@inline function gr_luminosity_correction(vin,width)
 theta=atan(width*GR_MsunBH*GR_rgeo/GR_distance);pix=theta/(vin[3]/2.0)*theta/(vin[3]/2.0)
 pix*GR_distance*GR_distance*4.0*GR_PI*vin[4]
end
@inline function gr_getz1(V,vin,y)
 r=y[1];a=vin[1];L=V[6]
 elocal=-(r*r+a*sqrt(r))/(r*sqrt(r*r-3.0*r+2.0*a*sqrt(r)))+L/sqrt(r)/sqrt(r*r-3.0*r+2.0*a*sqrt(r))
 elocal/(-1.0)
end
function gr_getz2(V,vin,y)
 @inbounds begin
 a=vin[1];a2=V[3];L=V[6];rms=V[5];r=y[1];theta=y[2];pr=y[5];r2=r*r;twor=2.0*r;st=sin(theta);ct=cos(theta);cos2=ct*ct;sin2=st*st
 sigma=r2+a2*cos2;delta=r2-twor+a2;ssig=(r2+a2)*(r2+a2)-a2*delta*sin2;gtt=-(1.0-2.0*r/sigma);gtph=-2.0*a*r*sin2/sigma;grr=sigma/delta;gphph=ssig*sin2/sigma
 ut_k=(r*r+a*sqrt(r))/(r*sqrt(r*r-3.0*r+2.0*a*sqrt(r)));ur_k=0.0;uphi_k=1.0/(sqrt(r)*sqrt(r*r-3.0*r+2.0*a*sqrt(r)))
 if r<rms
  delta=r*r-2.0*r+a2;lambda=(rms*rms-2.0*a*sqrt(rms)+a2)/(sqrt(rms*rms*rms)-2.0*sqrt(rms)+a);gamma=sqrt(1-2.0/3.0/rms);h=(2.0*r-a*lambda)/delta
  ut_k=gamma*(1.0+2/r*(1.0+h));ur_k=-sqrt(2.0/3.0/rms)*sqrt((rms/r-1.0)^3.0);uphi_k=gamma/r/r*(lambda+a*h)
 end
 ut=ut_k;uphi=uphi_k;ur=ur_k;omega=uphi/ut;k0=-(gtt+omega*omega*gphph+2.0*omega*gtph);ut=sqrt((1.0+grr*ur*ur)/k0);uphi=omega*ut;elocal=-ut+L*uphi+pr*ur
 end
 elocal/(-1.0)
end
@inline function gr_setup!(V,vin,x,y,width)
 @inbounds begin
 V[1]=1000.0;V[2]=(GR_PI/180.0)*vin[2];V[3]=vin[1]*vin[1];V[4]=1.0+sqrt(1.0-V[3])+1e-5;V[5]=gr_ISCO(vin)
 step=2.0*width/(Int(vin[3])+1.0);V[8]=-width+step*(x+1.0);V[9]=-width+step*(y+1.0)
 end
 nothing
end
function gr_pixel!(out,vin,x,ypix,::Val{1})
 V=MVector{9,Float64}(undef);y=MVector{6,Float64}(undef);d=MVector{6,Float64}(undef);scale=MVector{6,Float64}(undef);last=MVector{6,Float64}(undef)
 gr_setup!(V,vin,x,ypix,55.0);gr_initial!(V,vin,y,d);htry=.5;hnext=0.0;redshift=0.0
 @inbounds while true
  for i in 1:6;last[i]=y[i];end
  gr_geodesic!(V,vin,y,d)
  for i in 1:6;scale[i]=abs(y[i])+abs(d[i]*htry)+1e-3;end
  hnext,hdid=gr_rk5!(V,vin,y,d,htry,1e14,scale)
  if y[1]<50.0&&y[1]>V[5]&&(last[2]-GR_PI/2.0)*(y[2]-GR_PI/2.0)<0
   redshift=1.0/gr_getz1(V,vin,y);break
  end
  if y[1]>V[1]&&d[1]>0;break;end
  if y[1]<V[4];break;end
  htry=hnext
 end
 @inbounds begin
 base=3*(8*ypix+x);out[base+1]=V[8];out[base+2]=V[9];out[base+3]=redshift
 end
 nothing
end
function gr_pixel!(out,vin,x,ypix,::Val{2})
 V=MVector{9,Float64}(undef);y=MVector{6,Float64}(undef);d=MVector{6,Float64}(undef);scale=MVector{6,Float64}(undef)
 gr_setup!(V,vin,x,ypix,10.0);gr_initial!(V,vin,y,d);htry=.5;hnext=0.0;dtau=0.0;intensity=0.0;lum=0.0;lcorr=gr_luminosity_correction(vin,10.0)
 @inbounds while true
  gr_geodesic!(V,vin,y,d)
  for i in 1:6;scale[i]=abs(y[i])+abs(d[i]*htry)+1e-3;end
  hnext,hdid=gr_rk5!(V,vin,y,d,htry,1e14,scale)
  if (y[1]>V[1]&&d[1]>0)||y[1]<V[4]
   lum=intensity*vin[4]*vin[4]*vin[4]*lcorr;break
  end
  r=y[1];theta=y[2]
  if r<500.0
   zzz=gr_getz2(V,vin,y);flocal=vin[4]*zzz;zc=r*cos(theta);rc=r*sin(theta);nth=3e7*exp(-zc*zc/2.0/rc/rc)*r^(-1.1);te=1.7e11*r^(-.84);b=sqrt(8.0*GR_PI*.1*nth*GR_mp*GR_Cc*GR_Cc/6.0/r)
   vb=GR_e*b/2.0/GR_PI/GR_me/GR_Cc;theta_e=GR_kB*te/GR_me/GR_Cc/GR_Cc;v=flocal;xx=2.0*v/3.0/vb/theta_e/theta_e;kv=gr_K2(theta_e)
   c1=4.0*GR_PI*nth*GR_e*GR_e*v/sqrt(3.0)/kv/GR_Cc;c2=4.0505/xx^(1.0/6.0)*(1.0+.4/xx^(.25)+.5316/sqrt(xx))*exp(-1.8899*xx^(1.0/3.0));jnu=c1*c2;bnu=2.0*v*v*v*GR_h/GR_Cc/GR_Cc/(exp(GR_h*v/GR_kB/te)-1.0)
   # Scoped native correction: quadrature uses the accepted affine step hdid, not rejected htry.
   ds=hdid;dtau+=ds*GR_MsunBH*GR_rgeo*jnu/bnu*zzz;intensity+=ds*GR_MsunBH*GR_rgeo*jnu/flocal/flocal/flocal*exp(-dtau)*zzz
  end
  htry=hnext
 end
 @inbounds begin
 base=3*(8*ypix+x);out[base+1]=V[8];out[base+2]=V[9];out[base+3]=lum
 end
 nothing
end
