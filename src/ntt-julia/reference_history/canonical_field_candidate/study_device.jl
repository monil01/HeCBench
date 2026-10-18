
const PRIME=UInt64(0xffffffff00000001)
@inline function addp(x::UInt64,y::UInt64)
 r=x+y;r<x&&(r+=UInt64(0xffffffff));r>=PRIME&&(r-=PRIME);r
end
@inline function subp(x::UInt64,y::UInt64)
 r=x-y;r>x&&(r-=UInt64(0xffffffff));r
end
# Canonical modular shifts use the same field operation, with a different arithmetic lowering.
# Each bit doubling is overflow-safe, not UInt128 or foreign CUDA device interoperation.
@inline function lsp(x::UInt64,l::Int32)
 r=x>=PRIME ? x-PRIME : x
 for j in Int32(1):l;r=addp(r,r);end
 r
end
@inline function mulp(x::UInt64,y::UInt64)
 x>=PRIME&&(x%=PRIME);y>=PRIME&&(y%=PRIME)
 (x==0||y==0)&&return UInt64(0)
 (x|y)<UInt64(0xffffffff)&&return (x*y)%PRIME
 if x<y;x,y=y,x;end
 r=UInt64(0)
 for i in Int32(63):-Int32(1):Int32(0)
  r=r>UInt64(0x7fffffffffffffff) ? PRIME-((PRIME-r)<<1) : r<<1
  if ((y>>i)&UInt64(1))!=0;r=r>typemax(UInt64)-x ? r+(x-PRIME) : r+x;end
  r%=PRIME
 end
 r
end
@inline function butterfly8(a)
 s=(addp(a[1],a[5]),subp(a[1],a[5]),addp(a[3],a[7]),subp(a[3],a[7]),addp(a[2],a[6]),subp(a[2],a[6]),addp(a[4],a[8]),subp(a[4],a[8]))
 t=lsp(s[4],Int32(48));u=lsp(s[8],Int32(48))
 x=(addp(s[1],s[3]),addp(s[2],t),subp(s[1],s[3]),subp(s[2],t),addp(s[5],s[7]),addp(s[6],u),subp(s[5],s[7]),subp(s[6],u))
 a1=lsp(x[6],Int32(24));a2=lsp(x[7],Int32(48));a3=lsp(x[8],Int32(72))
 (addp(x[1],x[5]),addp(x[2],a1),addp(x[3],a2),addp(x[4],a3),subp(x[1],x[5]),subp(x[2],a1),subp(x[3],a2),subp(x[4],a3))
end
@inline function phase1!(x,buffer,b::Int32,t::Int32)
 f=(b<<9)|((t&Int32(0x3e))<<3)|(t&Int32(1))
 a=ntuple(i->(@inbounds x[(f|((Int32(i)-1)<<1))+17]),Val(8));z=butterfly8(a)
 for i in Int32(0):Int32(7);@inbounds buffer[((t<<3)|i)+1]=lsp(z[i+1],((t&Int32(1))<<2)*i*Int32(3));end
 nothing
end
@inline function phase2!(y,buffer,b::Int32,t::Int32)
 f=((t&Int32(0x38))<<3)|(t&Int32(7));tm=(b<<9)|f
 for i in Int32(0):Int32(7)
  j=i&~Int32(1);a=@inbounds buffer[(f|(j<<3))+1];v=@inbounds buffer[(f|((j+1)<<3))+1]
  s=iszero(i&Int32(1)) ? addp(a,v) : subp(a,v);idx=(((tm|(i<<3))&Int32(15))<<12)|((tm|(i<<3))>>4)
  @inbounds y[idx+17]=mulp(s,UInt64(18446462594437939201))%UInt32
 end
 nothing
end
