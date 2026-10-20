# Own full translation of native interval arithmetic and both stack Newton variants.
# Deliberately preserves source nearest rounding; no outward-bound guarantee.
using StaticArrays
struct IV
 lo::Float64
 hi::Float64
end
@inline IV(x::Float64)=IV(x,x)
@inline izero()=IV(NaN,NaN)
@inline iempty(x)=isnan(x.lo+x.hi)
@inline imin(x,y)=isnan(x) ? y : isnan(y) ? x : min(x,y)
@inline imax(x,y)=isnan(x) ? y : isnan(y) ? x : max(x,y)
@inline iwidth(x)=iempty(x) ? 0.0 : x.hi-x.lo
@inline imid(x)=(x.lo+x.hi)*.5
@inline iadd(x,y)=IV(x.lo+y.lo,x.hi+y.hi)
@inline isub(x,y)=IV(x.lo-y.hi,x.hi-y.lo)
@inline imul(x,y)=IV(imin(imin(x.lo*y.lo,x.lo*y.hi),imin(x.hi*y.lo,x.hi*y.hi)),imax(imax(x.lo*y.lo,x.lo*y.hi),imax(x.hi*y.lo,x.hi*y.hi)))
@inline function isquare(x)
 x.lo>=0 ? IV(x.lo*x.lo,x.hi*x.hi) : x.hi<=0 ? IV(x.hi*x.hi,x.lo*x.lo) : IV(0.,imax(x.lo*x.lo,x.hi*x.hi))
end
@inline function iintersect(x,y)
 lo=imax(x.lo,y.lo);hi=imin(x.hi,y.hi);lo<=hi ? IV(lo,hi) : izero()
end
@inline function idiv_nonzero(x,y)
 xl=y.hi<0 ? x.hi : x.lo;xu=y.hi<0 ? x.lo : x.hi
 if x.hi<0;yl=y.lo;yu=y.hi
 elseif x.lo<0;yl=y.hi<0 ? y.hi : y.lo;yu=yl
 else;yl=y.hi;yu=y.lo
 end
 IV(xl/yl,xu/yu)
end
@inline function idiv_positive(x,yu)
 x.lo==0&&x.hi==0&&return x
 x.hi<0 ? IV(-Inf,x.hi/yu) : x.lo<0 ? IV(-Inf,Inf) : IV(x.lo/yu,Inf)
end
@inline function idiv_negative(x,yl)
 x.lo==0&&x.hi==0&&return x
 x.hi<0 ? IV(x.hi/yl,Inf) : x.lo<0 ? IV(-Inf,Inf) : IV(-Inf,x.lo/yl)
end
@inline function idiv_first(x,y)
 if y.lo<=0&&y.hi>=0
  if y.lo!=0
   if y.hi!=0
    x.lo==0&&x.hi==0&&return x,false
    return x.hi<0 ? (IV(-Inf,x.hi/y.hi),true) : x.lo<0 ? (IV(-Inf,Inf),false) : (IV(-Inf,x.lo/y.lo),true)
   else;return idiv_negative(x,y.lo),false
   end
  elseif y.hi!=0;return idiv_positive(x,y.hi),false
  else;return izero(),false
  end
 else;return idiv_nonzero(x,y),false
 end
end
@inline idiv_second(x,y)=x.hi<0 ? IV(x.hi/y.lo,Inf) : IV(x.lo/y.hi,Inf)
@inline ifun(x,tid)=iadd(isquare(isub(x,IV(1.))),imul(IV(-Float64(tid)/65536.),x))
@inline ideriv(x,tid)=iadd(imul(IV(2.),x),IV(-Float64(tid)/65536. - 2.))
@inline iminimal(x,tid)=!iempty(x)&&(iwidth(x)<=Float64(1f-6)*abs(imid(x))||iwidth(ifun(x,tid))<=Float64(1f-6))
@inline ibisection(x,p1,p2)=(!iempty(p1)&&iwidth(p1)>Float64(.99f0)*iwidth(x))||(!iempty(p2)&&iwidth(p2)>Float64(.99f0)*iwidth(x))
@inline function irefine(ix,tid)
 x=imid(ix);q=ifun(IV(x),tid);d=ideriv(ix,tid);p1,has=idiv_first(q,d);p1=iintersect(isub(IV(x),p1),ix);p2=has ? iintersect(isub(IV(x),idiv_second(q,d)),ix) : izero();p1,p2,has,x
end
function interval_body!(tid,out,counts,domain,choice)
 work=MVector{128,IV}(undef);tos=Int32(0);nr=Int32(0)
 if choice==0
  tos=Int32(1);@inbounds work[1]=domain
  while tos>0
   ix=@inbounds work[tos];tos-=Int32(1)
   if iminimal(ix,tid)
    @inbounds out[nr*Int32(65536)+tid+Int32(1)]=ix;nr+=Int32(1)
   else
    p1,p2,has,x=irefine(ix,tid)
    if ibisection(ix,p1,p2);p1=IV(ix.lo,x);p2=IV(x,ix.hi);has=true;end
    if !iempty(p1);tos+=Int32(1);@inbounds work[tos]=p1;end
    if has&&!iempty(p2);tos+=Int32(1);@inbounds work[tos]=p2;end
   end
  end
 else
  ix=domain
  while true
   p1,p2,has,x=irefine(ix,tid)
   if iminimal(p1,tid);@inbounds out[nr*Int32(65536)+tid+Int32(1)]=p1;nr+=Int32(1);p1=izero();end
   if has&&iminimal(p2,tid);@inbounds out[nr*Int32(65536)+tid+Int32(1)]=p2;nr+=Int32(1);p2=izero();end
   if ibisection(ix,p1,p2);p1=IV(ix.lo,x);p2=IV(x,ix.hi);has=true;end
   if !iempty(p1)
    ix=p1
    if has&&!iempty(p2);tos+=Int32(1);@inbounds work[tos]=p2;end
   elseif has&&!iempty(p2);ix=p2
   elseif tos==0;break
   else;ix=@inbounds work[tos];tos-=Int32(1)
   end
  end
 end
 @inbounds counts[tid+Int32(1)]=nr
 nothing
end
@inline function interval_echo_body!(tid,echo,domain)
 a=ifun(IV(1.),tid);@inbounds echo[Int32(2)*tid+Int32(1)]=a.lo;@inbounds echo[Int32(2)*tid+Int32(2)]=a.hi
 if tid==0
  @inbounds echo[131073]=domain.lo;@inbounds echo[131074]=domain.hi;@inbounds echo[131075]=1. + 2.0^-53;@inbounds echo[131076]=1. + 3. * 2.0^-53
 end
 nothing
end
