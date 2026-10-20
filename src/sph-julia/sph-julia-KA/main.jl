using CUDA, KernelAbstractions
struct SPHParams
 rho::Float64; mass::Float64; spacing::Float64; h::Float64; g::Float64; dt::Float64; alpha::Float64; sigma::Float64; cs::Float64
 np::Int32; nf::Int32; nb::Int32; steps::Int32; frame::Int32
end
@assert sizeof(SPHParams)==96
@inline function pos(f,i)
 k=14*(i-1); @inbounds (f[k+3],f[k+4],f[k+5])
end
@inline function vel(f,i)
 k=14*(i-1); @inbounds (f[k+6],f[k+7],f[k+8])
end
@inline sub(p,q)=(p[1]-q[1],p[2]-q[2],p[3]-q[3])
@inline norm2(d)=d[1]*d[1]+d[2]*d[2]+d[3]*d[3]
@inline function smoothW(p,q,h)
 r=sqrt(norm2(sub(p,q))); C=1.0/(Float64(pi)*h*h*h);u=r/h;v=0.0
 if u<1.0;v=1.0-(3.0/2.0)*u*u+(3.0/4.0)*u*u*u
 elseif u>=1.0 && u<2.0;v=(1.0/4.0)*(2.0-u)^3.0
 end
 v*C
end
@inline function gradW(p,q,h)
 r=sqrt(norm2(sub(p,q)));C=1.0/(Float64(pi)*h*h*h);u=r/h;v=0.0
 if u<1.0;v=-1.0/(h*h)*(3.0-9.0/4.0*u)
 elseif u>=1.0 && u<2.0;v=-3.0/(4.0*h*r)*(2.0-u)^2.0
 end
 v*C
end
@inline function boundaryGamma(p,k,n,h,cs)
 d=sub(p,k);r=sqrt(norm2(d));y=sqrt(d[1]*d[1]*(n[1]*n[1])+d[2]*d[2]*(n[2]*n[2])+d[3]*d[3]*(n[3]*n[3]));x=r-y;u=y/h
 xi=(1.0-x/h)!=0.0 ? Float64(x<h) : 0.0
 C=xi*2.0*0.02*cs*cs/y;v=0.0
 if u>0.0 && u<2.0/3.0;v=2.0/3.0
 elseif u<1.0 && u>2.0/3.0;v=2.0*u-3.0/2.0*u*u
 elseif u<2.0 && u>1.0;v=0.5*(2.0-u)*(2.0-u)
 end
 v*C
end
@inline function density_increment(p,pv,q,qv,s)
 d=sub(p,q);v=sub(pv,qv);c=s.mass*gradW(p,q,s.h)
 (c*v[1]*d[1]+c*v[2]*d[2]+c*v[3]*d[3])*s.dt
end
@inline function acceleration(p,pv,prho,pp,q,qv,qrho,qp,s)
 d=sub(p,q);c=(pp/(prho*prho)+qp/(qrho*qrho))*s.mass*gradW(p,q,s.h)
 ax=-c*d[1];ay=-c*d[2];az=-c*d[3];v=sub(pv,qv);vd=v[1]*d[1]+v[2]*d[2]+v[3]*d[3]
 if vd<0.0
  nu=2.0*s.alpha*s.h*s.cs/(prho+qrho);eps=s.h/10.0;stress=nu*vd/(norm2(d)+eps*s.h*s.h);c=s.mass*stress*gradW(p,q,s.h)
  ax+=c*d[1];ay+=c*d[2];az+=c*d[3]
 end
 c=s.sigma*smoothW(p,q,s.h);(ax+c*d[1],ay+c*d[2],az+c*d[3])
end
@inline function pressure_body!(i,f,p)
 s=@inbounds p[1]
 if i<=s.nf
  k=14*(i-1);pp=pos(f,i);pv=vel(f,i);rho=@inbounds f[k+1]
  for j=1:Int(s.nf);rho+=density_increment(pp,pv,pos(f,j),vel(f,j),s);end
  B=s.rho*s.cs*s.cs/7.0
  @inbounds f[k+1]=rho
  @inbounds f[k+2]=B*((rho/s.rho)^7.0-1.0)
 end
 nothing
end
@inline function fluid_body!(i,f,p)
 s=@inbounds p[1]
 if i<=s.nf
  k=14*(i-1);pp=pos(f,i);pv=vel(f,i);rho=@inbounds f[k+1];pres=@inbounds f[k+2];ax=0.0;ay=0.0;az=-9.8
  for j=1:Int(s.nf)
   if i!=j
    qk=14*(j-1);qr=@inbounds f[qk+1];qp=@inbounds f[qk+2];a=acceleration(pp,pv,rho,pres,pos(f,j),vel(f,j),qr,qp,s);ax+=a[1];ay+=a[2];az+=a[3]
   end
  end
  @inbounds f[k+12]=ax
  @inbounds f[k+13]=ay
  @inbounds f[k+14]=az
 end
 nothing
end
@inline function boundary_body!(i,f,b,p)
 s=@inbounds p[1]
 if i<=s.nf
  k=14*(i-1);pp=pos(f,i);ax=@inbounds f[k+12];ay=@inbounds f[k+13];az=@inbounds f[k+14]
  for j=1:Int(s.nb)
   q=6*(j-1);kp=@inbounds (b[q+1],b[q+2],b[q+3]);kn=@inbounds (b[q+4],b[q+5],b[q+6]);c=boundaryGamma(pp,kp,kn,s.h,s.cs);ax+=c*kn[1];ay+=c*kn[2];az+=c*kn[3]
  end
  @inbounds f[k+12]=ax
  @inbounds f[k+13]=ay
  @inbounds f[k+14]=az
 end
 nothing
end
@inline function position_body!(i,f,p)
 s=@inbounds p[1]
 if i<=s.nf
  k=14*(i-1)
  for c=1:3
   a=@inbounds f[k+11+c];vh=@inbounds f[k+8+c];x=@inbounds f[k+2+c];vh=vh+s.dt*a
   @inbounds f[k+8+c]=vh
   @inbounds f[k+5+c]=vh+a*(s.dt/2.0)
   @inbounds f[k+2+c]=x+s.dt*vh
  end
 end
 nothing
end
@kernel function pressure_kernel!(f,@Const(p))
 i=@index(Global,Linear);pressure_body!(i,f,p)
end
@kernel function fluid_kernel!(f,@Const(p))
 i=@index(Global,Linear);fluid_body!(i,f,p)
end
@kernel function boundary_kernel!(f,@Const(b),@Const(p))
 i=@index(Global,Linear);boundary_body!(i,f,b,p)
end
@kernel function position_kernel!(f,@Const(p))
 i=@index(Global,Linear);position_body!(i,f,p)
end
function launch!(f,b,p,s)
 backend=KernelAbstractions.get_backend(f);nf=cld(Int(s.nf),256)*256;nb=cld(Int(s.nb),256)*256
 pressure_kernel!(backend,256)(f,p;ndrange=nf)
 fluid_kernel!(backend,256)(f,p;ndrange=nf)
 boundary_kernel!(backend,256)(f,b,p;ndrange=nb)
 position_kernel!(backend,256)(f,p;ndrange=nf)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main()
