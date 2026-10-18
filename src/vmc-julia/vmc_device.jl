@inline function vmc_random(s::UInt32)
 s=(UInt32(26757677)*s+UInt32(1))&UInt32(0x7fffffff)
 s,Float32(s)/2147483648f0
end
@inline vmc_dist(x,y,z)=sqrt(muladd(z,z,muladd(y,y,x*x)))
@inline function vmc_wave(x,y,z,u,v,w)
 r1=vmc_dist(x,y,z);r2=vmc_dist(u,v,w);r12=vmc_dist(x-u,y-v,z-w)
 muladd(.5f0,r12,1f0)*exp(-2f0*(r1+r2))
end
@inline function init_site!(i,seed,states)
 @inbounds states[i]=(UInt32(26757677)*(seed⊻UInt32(i-1))+UInt32(1))&UInt32(0x7fffffff)
 nothing
end
@inline function initialize_site!(i,a,states)
 @inbounds begin
 s=states[i];s,r=vmc_random(s);x=(r-.5f0)*4f0;s,r=vmc_random(s);y=(r-.5f0)*4f0;s,r=vmc_random(s);z=(r-.5f0)*4f0;s,r=vmc_random(s);u=(r-.5f0)*4f0;s,r=vmc_random(s);v=(r-.5f0)*4f0;s,r=vmc_random(s);w=(r-.5f0)*4f0
 a[1][i]=x;a[2][i]=y;a[3][i]=z;a[4][i]=u;a[5][i]=v;a[6][i]=w;a[7][i]=vmc_wave(x,y,z,u,v,w);states[i]=s
 end
 nothing
end
@inline function zero_site!(i,stats,N)
 @inbounds for j in Int32(0):Int32(3);stats[j*N+i]=0f0;end
 nothing
end
@inline function propagate_site!(i,steps,a,stats,states,N)
 @inbounds begin
 x=a[1][i];y=a[2][i];z=a[3][i];u=a[4][i];v=a[5][i];w=a[6][i];p=a[7][i];s=states[i]
 for step in Int32(1):steps
 s,r=vmc_random(s);xx=muladd(r-.5f0,2f0,x);s,r=vmc_random(s);yy=muladd(r-.5f0,2f0,y);s,r=vmc_random(s);zz=muladd(r-.5f0,2f0,z);s,r=vmc_random(s);uu=muladd(r-.5f0,2f0,u);s,r=vmc_random(s);vv=muladd(r-.5f0,2f0,v);s,r=vmc_random(s);ww=muladd(r-.5f0,2f0,w)
 pn=vmc_wave(xx,yy,zz,uu,vv,ww);s,r=vmc_random(s)
 if pn*pn>p*p*r
 stats[3*N+i]+=1f0;p=pn;x=xx;y=yy;z=zz;u=uu;v=vv;w=ww
 end
 stats[i]+=vmc_dist(x,y,z);stats[N+i]+=vmc_dist(u,v,w);stats[2*N+i]+=vmc_dist(x-u,y-v,z-w)
 end
 a[1][i]=x;a[2][i]=y;a[3][i]=z;a[4][i]=u;a[5][i]=v;a[6][i]=w;a[7][i]=p;states[i]=s
 end
 nothing
end
