# Source-derived TGV mathematical bodies. Original MIT Copyright Dong Wang.
# Each method is a source-kernel index body; frameworks provide their own GPU entry/loop.
@inline cdiv(z::ComplexF32, d::Float32) = z * inv(d)
@inline function point!(::Val{:scaledata},i,a,n,f)
 @inbounds a[i+1] *= f
end
@inline function point!(::Val{:arrayadd},i,c,a,b,n,x,y)
 @inbounds c[i+1] = x*a[i+1]+y*b[i+1]
end
@inline function point!(::Val{:arrayadd_real},i,c,a,b,n,x,y)
 @inbounds c[i+1] = x*a[i+1]+y*b[i+1]
end
@inline function point!(::Val{:arraydot},i,c,a,b,n)
 @inbounds c[i+1]=a[i+1]*b[i+1]
end
@inline function point!(::Val{:arrayreal},i,c,a,n)
 @inbounds c[i+1]=ComplexF32(max(real(a[i+1]),0f0),0f0)
end
@inline function point!(::Val{:update_r},i,r,tmp,b,n,sigma)
 @inbounds begin
  tmp[i+1] -= b[i+1]
  r[i+1] = cdiv(r[i+1]+sigma*tmp[i+1],1f0+sigma)
 end
end
@inline function point!(::Val{:grad_xx},i,o,a,n,r,c,t)
 j=rem(i,r*(c-1))+div(i,r*(c-1))*(r*c)
 @inbounds o[j+1]=a[j+r+1]-a[j+1]
end
@inline function point!(::Val{:grad_xx_bound},i,o,n,r,c,t)
 j=rem(i,r)+div(i,r)*(r*c)+r*(c-1)
 @inbounds o[j+1]=ComplexF32(0,0)
end
@inline function point!(::Val{:grad_yy},i,o,a,n,r,c,t)
 j=rem(i,r-1)+div(i,r-1)*r
 @inbounds o[j+1]=a[j+2]-a[j+1]
end
@inline function point!(::Val{:grad_yy_bound},i,o,n,r,c,t)
 j=i*r+(r-1)
 @inbounds o[j+1]=ComplexF32(0,0)
end
@inline function point!(::Val{:grad_tt},i,o,a,n,r,c,t,mu)
 @inbounds o[i+1]=cdiv(a[i+r*c+1]-a[i+1],mu)
end
@inline function point!(::Val{:grad_tt_bound},i,o,n,r,c,t)
 @inbounds o[i+(t-1)*r*c+1]=ComplexF32(0,0)
end
@inline function point!(::Val{:proj_p},i,p,tmp,wb,n,sigma,alpha)
 @inbounds begin
  p[i+1]-=sigma*(tmp[i+1]+wb[i+1]);p[i+n+1]-=sigma*(tmp[i+n+1]+wb[i+n+1]);p[i+2n+1]-=sigma*(tmp[i+2n+1]+wb[i+2n+1])
  s=sqrt(abs(p[i+1])*abs(p[i+1])+abs(p[i+n+1])*abs(p[i+n+1])+abs(p[i+2n+1])*abs(p[i+2n+1]));d=max(1f0,s/alpha)
  p[i+1]=cdiv(p[i+1],d);p[i+n+1]=cdiv(p[i+n+1],d);p[i+2n+1]=cdiv(p[i+2n+1],d)
 end
end
@inline function point!(::Val{:grad_xx_adj},i,o,a,n,r,c,t)
 j=rem(i,r*(c-2))+div(i,r*(c-2))*(r*c)+r
 @inbounds o[j+1]=a[j+1]-a[j-r+1]
end
@inline function point!(::Val{:grad_xx_adj_zero},i,o,a,r,c,t)
 j=rem(i,r)+div(i,r)*(r*c)
 @inbounds o[j+1]=a[j+1]
end
@inline function point!(::Val{:grad_xx_adj_bound},i,o,a,r,c,t)
 j=rem(i,r)+div(i,r)*(r*c)+r*(c-1)
 @inbounds o[j+1]=-1f0*a[j-r+1]
end
@inline function point!(::Val{:grad_yy_adj},i,o,a,n,r,c,t)
 j=rem(i,r-2)+div(i,r-2)*r+1
 @inbounds o[j+1]=a[j+1]-a[j]
end
@inline function point!(::Val{:grad_yy_adj_zero},i,o,a,r,c,t)
 j=i*r
 @inbounds o[j+1]=a[j+1]
end
@inline function point!(::Val{:grad_yy_adj_bound},i,o,a,r,c,t)
 j=i*r+(r-1)
 @inbounds o[j+1]=-1f0*a[j]
end
@inline function point!(::Val{:grad_tt_adj},i,o,a,n,r,c,t,mu)
 stride=r*c
 @inbounds o[i+stride+1]=cdiv(a[i+stride+1]-a[i+1],mu)
end
@inline function point!(::Val{:grad_tt_adj_zero},i,o,a,r,c,t)
 @inbounds o[i+1]=a[i+1]
end
@inline function point!(::Val{:grad_tt_adj_bound},i,o,a,r,c,t)
 j=i+r*c*(t-1)
 @inbounds o[j+1]=-1f0*a[j-r*c+1]
end
@inline function point!(::Val{:proj_q},i,q,tmp,n,sigma,alpha)
 @inbounds begin
  tmp[i+1]=tmp[i+3n+1];tmp[i+n+1]=tmp[i+4n+1];tmp[i+2n+1]=tmp[i+5n+1]
  tmp[i+3n+1]=cdiv(tmp[i+6n+1]+tmp[i+7n+1],2f0);tmp[i+4n+1]=cdiv(tmp[i+8n+1]+tmp[i+9n+1],2f0);tmp[i+5n+1]=cdiv(tmp[i+10n+1]+tmp[i+11n+1],2f0)
  for k=0:5;q[i+k*n+1]-=sigma*tmp[i+k*n+1];end
  x0=abs(q[i+1]);x1=abs(q[i+n+1]);x2=abs(q[i+2n+1]);x3=abs(q[i+3n+1]);x4=abs(q[i+4n+1]);x5=abs(q[i+5n+1]);s=sqrt(x0*x0+x1*x1+x2*x2+2*x3*x3+2*x4*x4+2*x5*x5);d=max(1f0,s/alpha)
  for k=0:5;q[i+k*n+1]=cdiv(q[i+k*n+1],d);end
 end
end
@inline function point!(::Val{:update_s},i,s,tmp,z,n,tau)
 @inbounds begin;v=tmp[i+1]+tmp[i+n+1]+tmp[i+2n+1];s[i+1]-=tau*(z[i+1]+v);end
end
@inline function point!(::Val{:update_w},i,w,tmp,p,n,tau)
 @inbounds begin
  v=tmp[i+3n+1]+tmp[i+4n+1]+tmp[i+5n+1];w[i+1]-=tau*(v-p[i+1])
  v=tmp[i+6n+1]+tmp[i+7n+1]+tmp[i+8n+1];w[i+n+1]-=tau*(v-p[i+n+1])
  v=tmp[i+9n+1]+tmp[i+10n+1]+tmp[i+11n+1];w[i+2n+1]-=tau*(v-p[i+2n+1])
 end
end
@inline function point!(::Val{:shrink},i,s2,s,beta,n)
 @inbounds begin;s[i+1]=max(s[i+1]-beta,0f0);s2[i+1]=ComplexF32(s[i+1],0f0);end
end
@inline function point!(::Val{:arrayabs},i,o,a,n)
 @inbounds o[i+1]=abs(a[i+1])
end
