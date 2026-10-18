# Source WIMAX Z96, CW2: actual four-byte native h_element physical layout.
struct HElement
 x::Int8
 y::Int8
 value::Int8
 valid::Int8
end
const LZ=Int32(96);const LROW=Int32(1152);const LCOL=Int32(2304);const LMESSAGE=Int32(27648)
@inline function cnp_prepare!(::Val{FIRST},sx,sy,br,mc,llr,R,count,compact,cache) where FIRST
 cw=mc*Int32(2)+sy;offset=LMESSAGE*cw+br*LZ+sx;s=Int32(count[br+1]);line=sy*LZ+sx;sign=1f0;min1=1000f0;min2=1000f0;idx=Int32(0);bits=UInt32(0)
 @inbounds for k in Int32(0):s-Int32(1)
  he=compact[k*Int32(12)+br+1];col=Int32(he.y);shift=sx+Int32(he.value);if shift>=LZ;shift-=LZ;end
  q=llr[cw*LCOL+col*LZ+shift+1]
  if !FIRST
   old=R[offset+col*LROW+1];cache[k*Int32(192)+line+1]=old;q-=old
  end
  neg=q<0f0;sign*=neg ? -1f0 : 1f0;bits|=UInt32(neg)<<k;v=abs(q)
  if v<min1;min2=min1;min1=v;idx=k
  elseif v<min2;min2=v
  end
 end
 (sign,min1,min2,idx,bits,offset,line,s)
end
@inline function cnp_finish!(::Val{FIRST},br,dt,R,compact,cache,localstate) where FIRST
 sign,min1,min2,idx,bits,offset,line,s=localstate
 @inbounds for k in Int32(0):s-Int32(1)
  sg=(((bits>>k)&UInt32(1))==0) ? 1f0 : -1f0;value=(.75f0*sign)*sg*(k!=idx ? min1 : min2);he=compact[k*Int32(12)+br+1];addr=offset+Int32(he.y)*LROW+1
  dt[addr]=FIRST ? value : value-cache[k*Int32(192)+line+1];R[addr]=value
 end
 nothing
end
@inline function vnp_site!(::Val{LAST},sx,sy,bc,mc,llr,dt,hd,count,compact) where LAST
 cw=mc*Int32(2)+sy;var=cw*LCOL+bc*LZ+sx;offset=LMESSAGE*cw+bc*LROW
 @inbounds begin
  app=llr[var+1]
  for k in Int32(0):Int32(count[bc+1])-Int32(1)
   he=compact[k*Int32(24)+bc+1];shift=sx-Int32(he.value);if shift<0;shift+=LZ;end
   row=Int32(he.x)*LZ+shift;app+=dt[offset+row+1]
  end
  if LAST;hd[var+1]=app>0f0 ? Int32(0) : Int32(1)
  else;llr[var+1]=app
  end
 end
 nothing
end
