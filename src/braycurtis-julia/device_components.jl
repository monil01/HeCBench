# Pure scalar source-device arithmetic; exact source extraction/provenance stored separately.
@inline bray_sign(x)=ifelse(x>zero(x),one(x),ifelse(x<zero(x),-one(x),zero(x)))
@inline function bray_pair(i,c,a,b)
 j=i%c.n;row=(i÷c.n)%c.m;batch=i÷(c.n*c.m);T=eltype(a);d=zero(T);s=zero(T)
 for k in Int32(0):c.k-Int32(1)
  @inbounds x=a[(batch*c.m+row)*c.k+k+Int32(1)];y=b[(batch*c.n+j)*c.k+k+Int32(1)]
  d+=abs(x-y);s+=abs(x+y)
 end
 d,s
end
@inline function bray_forward(i::Int32,stride::Int32,c,a,b,go,out,sd,ss,g1,g2)
 while i<c.nq
  d,s=bray_pair(i,c,a,b)
  @inbounds out[i+Int32(1)]=d/s
  i+=stride
 end
 nothing
end
@inline function bray_components(i::Int32,stride::Int32,c,a,b,go,out,sd,ss,g1,g2)
 while i<c.nq
  d,s=bray_pair(i,c,a,b)
  @inbounds sd[i+Int32(1)]=d;ss[i+Int32(1)]=s
  i+=stride
 end
 nothing
end
@inline function bray_grad1(i::Int32,stride::Int32,c,a,b,go,out,sd,ss,g1,g2)
 while i<c.na
  k=i%c.k;row=(i÷c.k)%c.m;batch=i÷(c.k*c.m);value=zero(eltype(a))
  for j in Int32(0):c.n-Int32(1)
   q=(batch*c.m+row)*c.n+j+Int32(1)
   @inbounds diff=a[i+Int32(1)]-b[(batch*c.n+j)*c.k+k+Int32(1)];sumv=a[i+Int32(1)]+b[(batch*c.n+j)*c.k+k+Int32(1)]
   @inbounds value+=go[q]*(bray_sign(diff)/ss[q]-bray_sign(sumv)*sd[q]/ss[q]/ss[q])
  end
  @inbounds g1[i+Int32(1)]=value
  i+=stride
 end
 nothing
end
@inline function bray_grad2(i::Int32,stride::Int32,c,a,b,go,out,sd,ss,g1,g2)
 while i<c.nb
  k=i%c.k;j=(i÷c.k)%c.n;batch=i÷(c.k*c.n);value=zero(eltype(a))
  for row in Int32(0):c.m-Int32(1)
   q=(batch*c.m+row)*c.n+j+Int32(1)
   @inbounds diff=a[(batch*c.m+row)*c.k+k+Int32(1)]-b[i+Int32(1)];sumv=a[(batch*c.m+row)*c.k+k+Int32(1)]+b[i+Int32(1)]
   @inbounds value+=go[q]*(-bray_sign(diff)/ss[q]-bray_sign(sumv)*sd[q]/ss[q]/ss[q])
  end
  @inbounds g2[i+Int32(1)]=value
  i+=stride
 end
 nothing
end
