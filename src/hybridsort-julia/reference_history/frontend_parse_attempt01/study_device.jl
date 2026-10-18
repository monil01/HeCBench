
# Literal source pipeline stages; zero-based device addresses translated at each load/store.
@inline function sort4(r)
 x,y,z,w=r;a=x>y ? y : x;b=y>x ? y : x;c=z>w ? w : z;d=w>z ? w : z
 x=a>c ? c : a;y=b>d ? d : b;z=c>a ? c : a;w=d>b ? d : b
 return (x,y>z ? z : y,z>y ? z : y,w)
end
@inline load4(a,i)=(a[4i+1],a[4i+2],a[4i+3],a[4i+4])
@inline function store4!(a,i,v)
 for j in 1:4;a[4i+j]=v[j];end
end
@inline function stage!(::Val{1},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 for i=lid:ls:3071;sh[i+1]=UInt32(0);end
 CUDA.sync_threads()
 for pos=gid:gs:n-1
  data=UInt32(trunc(Int32,((input[pos+1]-lo)/(hi-lo))*1024f0))
  CUDA.atomic_add!(pointer(sh,(lid÷32)*1024+Int(data&0x3ff)+1),UInt32(1))
 end
 CUDA.sync_threads()
 for pos=lid:ls:1023
  z=UInt32(0);for i=0:1024:2048;z+=sh[pos+i+1]&UInt32(0x07ffffff);end
  CUDA.atomic_add!(pointer(offsets,pos+1),z)
 end
end
@inline function stage!(::Val{2},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 for i=lid:ls:1023;sh[i+1]=UInt32(0);end;CUDA.sync_threads()
 for tid=gid:gs:n-1
  elem=input[tid+1];idx=511;jump=256;piv=pivots[idx+1]
  while jump>=1;idx=elem<piv ? idx-jump : idx+jump;piv=pivots[idx+1];jump÷=2;end
  idx=elem<piv ? idx : idx+1
  old=CUDA.atomic_add!(pointer(sh,idx+1),UInt32(1));indices[tid+1]=Int32((old<<10)+UInt32(idx))
 end
 CUDA.sync_threads()
 for i=lid:ls:1023;prefix[bid*1024+i+1]=sh[i+1]&UInt32(0x07ffffff);end
end
@inline function stage!(::Val{3},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 z=UInt32(0)
 for i=gid:1024:size*1024-1;x=prefix[i+1];prefix[i+1]=z;z+=x;end
 offsets[gid+1]=z
end
@inline function stage!(::Val{4},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 for i=lid:ls:1023;sh[i+1]=offsets[i+1]+prefix[bid*1024+i+1];end;CUDA.sync_threads()
 for tid=gid:gs:n-1
  id=indices[tid+1];output[Int(sh[Int(id&1023)+1])+Int(id>>10)+1]=input[tid+1]
 end
end
@inline function stage!(::Val{5},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 if gid<size;store4!(output,gid,sort4(load4(input,gid)));end
end
@inline function stage!(::Val{6},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 division=gid÷tpd
 if division<1024
  astart=Int(starts[division+1])+(gid-division*tpd)*nr;bstart=astart+nr÷2;finish=Int(starts[division+2])
  if astart<finish
   if bstart>=finish
    for i=0:finish-astart-1;store4!(output,astart+i,load4(input,astart+i));end
   else
    ai=0;bi=0;oi=0;a=load4(input,astart);b=load4(input,bstart)
    while true
     na=load4(input,astart+ai+1);nb=bstart+bi+1>=size ? (0f0,0f0,0f0,0f0) : load4(input,bstart+bi+1)
     low=ntuple(j->a[j]<b[5-j] ? a[j] : b[5-j],4);high=ntuple(j->a[5-j]>=b[j] ? a[5-j] : b[j],4)
     a=sort4(low);b=sort4(high);store4!(output,astart+oi,a);oi+=1
     lefta=ai+1<nr÷2;leftb=bi+1<nr÷2 && bstart+bi+1<finish
     if lefta
      if leftb
       if na[1]<nb[1];ai+=1;a=na;else;bi+=1;a=nb;end
      else;ai+=1;a=na;end
     elseif leftb;bi+=1;a=nb
     else;break;end
    end
    store4!(output,astart+oi,b)
   end
  end
 end
end
@inline function stage!(::Val{7},gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 div=bid # gid is block-x global, bid is block-y division for this stage.
 if Int(finals[div+1])+gid<Int(finals[div+2])
  output[Int(finals[div+1])+gid+1]=input[Int(starts[div+1])*4+Int(nulls[div+1])+gid+1]
 end
end
