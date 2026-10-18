const PRIME=UInt64(0xffffffff00000001)
const MASK32=UInt64(0xffffffff)
@inline function high64(a::UInt64,b::UInt64)
 a0=a&MASK32;a1=a>>32;b0=b&MASK32;b1=b>>32
 w0=a0*b0;t=a1*b0+(w0>>32);w1=t&MASK32;w2=t>>32;w1=a0*b1+w1
 return a1*b1+w2+(w1>>32)
end
@inline function fmul(a::UInt64,b::UInt64)
 low=a*b;hi=high64(a,b);c=hi&MASK32;d=hi>>32
 under=low<d;v=low-d;v-=under ? MASK32 : UInt64(0)
 shift=(c<<32)-c;over=v>typemax(UInt64)-shift
 return v+shift+(over ? MASK32 : UInt64(0))
end
@inline function fadd(a::UInt64,b::UInt64)
 b=b>=PRIME ? b-PRIME : b;over=a>typemax(UInt64)-b;v=a+b;tmp=over ? MASK32 : UInt64(0)
 over2=v>typemax(UInt64)-tmp
 return v+tmp+(over2 ? MASK32 : UInt64(0))
end
@inline tmul(a,b)=ntuple(i->fmul(a[i],b[i]),Val(12))
@inline function accum(a)
 v0=fadd(fadd(a[1],a[2]),fadd(a[3],a[4]));v1=fadd(fadd(a[5],a[6]),fadd(a[7],a[8]));v2=fadd(fadd(a[9],a[10]),fadd(a[11],a[12]))
 return fadd(v2,fadd(v0,v1))
end
@inline function mds_apply(state,mds)
 return ntuple(Val(12)) do row
  v=ntuple(i->fmul(state[i],@inbounds(mds[12*(row-1)+i])),Val(12));accum(v)
 end
end
@inline function powacc(m,a,tail)
 for _ in 1:m;a=tmul(a,a);end
 return tmul(a,tail)
end
@inline function invsbox(state)
 t1=tmul(state,state);t2=tmul(t1,t1);t3=powacc(3,t2,t2);t4=powacc(6,t3,t3);t5=powacc(12,t4,t4);t6=powacc(6,t5,t3);t7=powacc(31,t6,t6)
 a=tmul(t6,tmul(t7,t7));a=tmul(a,a);a=tmul(a,a);b=tmul(tmul(t1,t2),state);return tmul(a,b)
end
@inline function rescue_merge!(input,ib,output,ob,mds,ark1,ark2)
 state=ntuple(i->i<=8 ? @inbounds(input[ib+i]) : i==12 ? UInt64(8) : UInt64(0),Val(12))
 for round in 0:6
  p2=tmul(state,state);p4=tmul(p2,p2);p6=tmul(p2,p4);state=mds_apply(tmul(state,p6),mds)
  state=let current=state,r=round
   ntuple(i->fadd(current[i],@inbounds(ark1[r*12+i])),Val(12))
  end
  state=mds_apply(invsbox(state),mds)
  state=let current=state,r=round
   ntuple(i->fadd(current[i],@inbounds(ark2[r*12+i])),Val(12))
  end
 end
 @inbounds for i in 1:4;output[ob+i]=state[i];end
 return nothing
end
