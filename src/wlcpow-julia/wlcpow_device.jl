struct WVec4
 x::Float32;y::Float32;z::Float32;w::Float32
end
struct WInt2
 x::Int32;y::Int32
end
@inline wminimum(d,p)=d+(d>-p*.5f0 ? (d<p*.5f0 ? 0f0 : -p) : p)
@inline function wgauss(pred,u,v)
 a=pred ? UInt32(u) : UInt32(v);b=pred ? UInt32(v) : UInt32(u);s=UInt32(0)
 for _ in 1:4
  s+=UInt32(0x9e3779b9);a+=((b<<4)+UInt32(0xa341316c))⊻(b+s)⊻((b>>5)+UInt32(0xc8013ea4));b+=((a<<4)+UInt32(0xad90777d))⊻(a+s)⊻((a>>5)+UInt32(0x7e95761e))
 end
 f=sinpi(Float32(reinterpret(Int32,a))*4.6566128730773926f-10);r=sqrt(-2f0*0.6931471805599453f0*log2(Float32(b)*2.3283064365386963f-10))
 min(4f0,max(-4f0,r*f))
end
@inline function wsite!(i,force,co,vel,nb,bonds,l0s,sh,period,N,nt)
 @inbounds begin
  x=co[i];v=vel[i];fx=0f0;fy=0f0;fz=0f0;sz=nt+Int32(1)
  for p in Int32(0):nb[i]-Int32(1)
   b=bonds[i+p];j=b.x+Int32(1);t=b.y+Int32(1);y=co[j];w=vel[j];dx=wminimum(x.x-y.x,period[1]);dy=wminimum(x.y-y.y,period[2]);dz=wminimum(x.z-y.z,period[3]);vx=v.x-w.x;vy=v.y-w.y;vz=v.z-w.z
   temp=sh[t];r0=sh[sz+t];mu0=sh[2*sz+t];qp=sh[3*sz+t];gc=sh[4*sz+t];gt=sh[5*sz+t];sc=sh[6*sz+t];st=sh[7*sz+t];l0=Float32(l0s[i+p]);ra=sqrt(dx*dx+dy*dy+dz*dz);lm=l0*r0;rr=1f0/r0;sr=(1f0-rr)*(1f0-rr);kp=l0^qp*temp*(.25f0/sr-.25f0+rr);mu=.433f0*(temp*(-.25f0/sr+.25f0+.5f0*rr/(sr*(1f0-rr)))/(lm*rr)+kp*(qp+1f0)/(l0^(qp+1f0)));lambda=mu/mu0;kp/=lambda;rr=ra/lm;power=ra^(qp+1f0);vv=(dx*vx+dy*vy+dz*vz)/ra
   if rr>=.99;rr=.99f0;end
   if power<.01;power=.01f0;end
   v1=reinterpret(Int32,v.w);v2=reinterpret(Int32,w.w);pred=v1>v2
   a=wgauss(pred,v1,v2);b01=wgauss(pred,v1,v2+Int32(1));b02=wgauss(pred,v1,v2+Int32(2));b10=wgauss(pred,v1+Int32(1),v2);b11=wgauss(pred,v1+Int32(1),v2+Int32(1));b12=wgauss(pred,v1+Int32(1),v2+Int32(2));b20=wgauss(pred,v1+Int32(2),v2);b21=wgauss(pred,v1+Int32(2),v2+Int32(1));b22=wgauss(pred,v1+Int32(2),v2+Int32(2));trace=(a+b11+b22)/3f0;wx=(a-trace)*dx+.5f0*(b01+b10)*dy+.5f0*(b02+b20)*dz;wy=.5f0*(b10+b01)*dx+(b11-trace)*dy+.5f0*(b12+b21)*dz;wz=.5f0*(b20+b02)*dx+.5f0*(b21+b12)*dy+(b22-trace)*dz
   ff=-temp*(.25f0/(1f0-rr)/(1f0-rr)-.25f0+rr)/lambda/ra+kp/power+(sc*trace-gc*vv)/ra;fx+=dx*ff-gt*vx+st*wx/ra;fy+=dy*ff-gt*vy+st*wy/ra;fz+=dz*ff-gt*vz+st*wz/ra
  end
  force[1][i]+=Float64(fx);force[2][i]+=Float64(fy);force[3][i]+=Float64(fz)
 end
 nothing
end
