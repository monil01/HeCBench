
@inline advance_seed(s::Int32)=rem(Int32(1103515245)*s+Int32(12345),typemax(Int32))
@inline function normal_pair(s::Int32)
 s=advance_seed(s);u=abs(Float32(s)/Float32(typemax(Int32)));s=advance_seed(s);v=abs(Float32(s)/Float32(typemax(Int32)))
 s,sqrt(-2f0*log(u))*cos((2f0*Float32(3.1415926535897932))*v)
end
@inline function motion!(i,n,ax,ay,xj,yj,seeds)
 if i<n
  j=i+17;s=@inbounds seeds[j];s,z=normal_pair(s);@inbounds ax[j]=xj[j]+(1f0+5f0*z);s,z=normal_pair(s);@inbounds ay[j]=yj[j]+(-2f0+2f0*z);@inbounds seeds[j]=s
 end
 nothing
end
@inline function likelihood!(i,n,k,ax,ay,ind,obj,like,im,w)
 if i<n
  j=i+17;x=trunc(Int32,@inbounds ax[j]);y=trunc(Int32,@inbounds ay[j]);acc=0f0
  for t in Int32(0):Int32(68)
   dx=@inbounds obj[2t+18];dy=@inbounds obj[2t+17];ix=abs((x+dx)*Int32(256)+(y+dy)*Int32(4)+k);ix>=Int32(16384)&&(ix=Int32(0));@inbounds ind[i*Int32(69)+t+17]=ix;pixel=Int32(@inbounds im[ix+17]);a=pixel-Int32(100);b=pixel-Int32(228);acc+=Float32(a*a-b*b)/50f0
  end
  ll=acc/69f0-300f0;@inbounds like[j]=ll;@inbounds w[j]=(1f0/Float32(n))*exp(ll)
 end
 nothing
end
@inline function sum_partials!(n,part)
 z=0f0;for j in Int32(0):cld(n,Int32(256))-Int32(1);z+=@inbounds part[j+17];end;@inbounds part[17]=z;nothing
end
@inline function scale!(i,n,w,part)
 if i<n;@inbounds w[i+17]=w[i+17]/part[17];end;nothing
end
@inline function cdf_offset!(n,w,cdf,u,seeds)
 @inbounds cdf[17]=w[17]
 for j in Int32(1):n-Int32(1);@inbounds cdf[j+17]=w[j+17]+cdf[j+16];end
 s,z=normal_pair(@inbounds seeds[17]);@inbounds seeds[17]=s;@inbounds u[17]=(1f0/Float32(n))*z;nothing
end
@inline function thresholds!(t,n,u,offset)
 for j in t:Int32(256):n-Int32(1);@inbounds u[j+17]=offset+Float32(j)/Float32(n);end;nothing
end
@inline function resample!(i,n,ax,ay,cdf,u,xj,yj)
 if i<n
  threshold=@inbounds u[i+17];chosen=n-Int32(1)
  for j in Int32(0):n-Int32(1)
   if(@inbounds cdf[j+17])>=threshold;chosen=j;break;end
  end
  @inbounds xj[i+17]=ax[chosen+17];@inbounds yj[i+17]=ay[chosen+17]
 end;nothing
end
