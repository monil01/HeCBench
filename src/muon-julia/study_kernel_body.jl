const NS_A=3.4445f0;const NS_B=-4.7750f0;const NS_C=2.0315f0
@inline encode_bf(x::Float32)=UInt16((reinterpret(UInt32,x)+UInt32(0x7fff)+((reinterpret(UInt32,x)>>16)&UInt32(1)))>>16)
@inline decode_bf(x::UInt16)=reinterpret(Float32,UInt32(x)<<16)
@inline function momentum_value!(mom,g,i,beta)
 t=(1f0-beta)*@inbounds(g[i]);mi=muladd(beta,@inbounds(mom[i]),t);@inbounds mom[i]=mi;return encode_bf(muladd(beta,mi,t))
end
@inline function momentum_body!(idx,mom,g,out,beta,n)
 n4=n>>2
 if idx<n4
  for lane in Int32(0):Int32(3);i=idx*Int32(4)+lane+Int32(1);@inbounds out[i]=momentum_value!(mom,g,i,beta);end
 end
 if idx==0
  for i in n4*Int32(4)+Int32(1):n;@inbounds out[i]=momentum_value!(mom,g,i,beta);end
 end
 return
end
@inline function vector_body!(idx,out,a,b,k1,k2,n,mode)
 n4=n>>2
 if idx<n4
  for lane in Int32(0):Int32(3)
   i=idx*Int32(4)+lane+Int32(1);x=decode_bf(@inbounds a[i]);v=mode==Int32(0) ? x*k1 : mode==Int32(1) ? muladd(k1,x,k2*decode_bf(@inbounds b[i])) : muladd(k1,x,decode_bf(@inbounds b[i]));@inbounds out[i]=encode_bf(v)
  end
 end
 if idx==0
  for i in n4*Int32(4)+Int32(1):n
   x=decode_bf(@inbounds a[i]);v=mode==Int32(0) ? x*k1 : mode==Int32(1) ? muladd(k1,x,k2*decode_bf(@inbounds b[i])) : muladd(k1,x,decode_bf(@inbounds b[i]));@inbounds out[i]=encode_bf(v)
  end
 end
 return
end
@inline function parameter_body!(idx,p,x,lr,wd,scale,n)
 decay=1f0-lr*wd;n4=n>>2
 if idx<n4
  for lane in Int32(0):Int32(3);i=idx*Int32(4)+lane+Int32(1);@inbounds p[i]=muladd(-lr,scale*decode_bf(x[i]),p[i]*decay);end
 end
 if idx==0
  for i in n4*Int32(4)+Int32(1):n;@inbounds p[i]=muladd(-lr,scale*decode_bf(x[i]),p[i]*decay);end
 end
 return
end
