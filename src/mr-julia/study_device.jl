const MR_MASK=(255,85,51,73,199,93,59,17,15,229,195,89,215,237,203,33,31,117,83,105,231,125,91,49,47,5,227,121,247,13,235,65,63,149,115,137,7,157,123,81,79,37,3,153,23,45,11,97,95,181,147,169,39,189,155,113,111,69,35,185,55,77,43,129,127,213,179,201,71,221,187,145,143,101,67,217,87,109,75,161,159,245,211,233,103,253,219,177,175,133,99,249,119,141,107,193,191,21,243,9,135,29,251,209,207,165,131,25,151,173,139,225,223,53,19,41,167,61,27,241,239,197,163,57,183,205,171,1,)
@inline function mont_prod(a::UInt32,b::UInt32,n::UInt32,npi::UInt32)
 t=UInt64(a)*UInt64(b);m=(t%UInt32)*npi;u=((t+UInt64(m)*UInt64(n))>>32)%UInt32
 if UInt64(u)<(t>>32);return u-n;end
 u>=n ? u-n : u
end
@inline function inverse32(a::UInt32)
 @inbounds ret=UInt32(MR_MASK[Int((a>>1)&UInt32(127))+1])
 ret*=UInt32(2)+a*ret;ret*=UInt32(2)+a*ret;ret
end
@inline function modn(n::UInt32)
 if n<=UInt32(1)<<31
  r=((UInt32(1)<<31)%n)<<1
  return r<n ? r : r-n
 end
 -n
end
@inline function efficient(bases,cnt,n::UInt32)
 npi=inverse32(n);r=modn(n);u=n-UInt32(1);nr=n-r;t=0
 while (u&UInt32(1))==0;t+=1;u>>=1;end
 for j in 1:cnt
  @inbounds a=bases[j]
  d=r;uc=u;A=((UInt64(a)<<32)%UInt64(n))%UInt32
  if A==0;continue;end
  while true
   if (uc&UInt32(1))!=0;d=mont_prod(d,A,n,npi);end
   A=mont_prod(A,A,n,npi);uc>>=1;if uc==0;break;end
  end
  if d==r||d==nr;continue;end
  i=1
  while i<t
   d=mont_prod(d,d,n,npi)
   if d==r;return Int32(0);end
   if d==nr;break;end
   i+=1
  end
  if i==t;return Int32(0);end
 end
 Int32(1)
end
@inline function powmod(a::UInt32,b::UInt32,n::UInt32)
 d=UInt64(1);A=UInt64(a)
 while true
  if (b&UInt32(1))!=0;d=(d*A)%UInt64(n);end
  A=(A*A)%UInt64(n);b>>=1;if b==0;break;end
 end
 d%UInt32
end
@inline function straightforward(bases,cnt,n::UInt32)
 u=n-UInt32(1);t=0
 while u%UInt32(2)==0;t+=1;u>>=1;end
 for j in 1:cnt
  @inbounds a=bases[j]
  if a>=n;a%=n;end
  if a==0;continue;end
  x=powmod(a,u,n)
  if x==1||x==n-UInt32(1);continue;end
  i=1
  while i<t
   x=((UInt64(x)*UInt64(x))%UInt64(n))%UInt32
   if x==1;return Int32(0);end
   if x==n-UInt32(1);break;end
   i+=1
  end
  if i==t;return Int32(0);end
 end
 Int32(1)
end
@inline mr_test(b,c,n,eff)=eff ? efficient(b,c,n) : straightforward(b,c,n)
