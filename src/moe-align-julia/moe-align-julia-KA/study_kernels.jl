using CUDA,Atomix,KernelAbstractions
@kernel function small_ka!(top,sorted,experts,total,mapping,E,N,maxpad,maxblocks,stride,usemap,::Val{Words}) where {Words}
 s=@localmem Int32 (Words,);tx=Int32(@index(Local,Linear))-Int32(1);tid=tx-Int32(256);base=E+Int32(1)
 if tx<Int32(256)
  for i in tx:Int32(256):maxpad-Int32(1);@inbounds sorted[i+1]=N;end
 else
  for e in Int32(0):E-Int32(1);@inbounds s[base+(tid+1)*E+e+1]=Int32(0);end
  for i in tid:stride:N-Int32(1)
   e=@inbounds top[i+1]
   if usemap;e=@inbounds mapping[e+1];if e==Int32(-1);continue;end;end
   @inbounds s[base+(tid+1)*E+e+1]+=Int32(1)
  end
 end
 @synchronize()
 if tx>=Int32(256)&&tid<E
  @inbounds s[base+tid+1]=Int32(0)
  for j in Int32(1):stride;@inbounds s[base+j*E+tid+1]+=s[base+(j-1)*E+tid+1];end
 end
 @synchronize()
 if tx==Int32(256)
  @inbounds s[1]=Int32(0)
  for e in Int32(1):E;@inbounds s[e+1]=s[e]+cld(s[base+stride*E+e],Int32(32))*Int32(32);end
  @inbounds total[1]=s[E+1]
 end
 @synchronize()
 if tx>=Int32(256)
  if tid<E
   for i in (@inbounds s[tid+1]):Int32(32):(@inbounds s[tid+2])-Int32(1);@inbounds experts[i÷Int32(32)+1]=tid;end
  end
  start=(@inbounds s[E+1])÷Int32(32)+tid
  for i in start:stride:maxblocks-Int32(1);@inbounds experts[i+1]=Int32(0);end
  for i in tid:stride:N-Int32(1)
   e=@inbounds top[i+1]
   if usemap;e=@inbounds mapping[e+1];if e==Int32(-1);continue;end;end
   rank=@inbounds s[base+tid*E+e+1]+s[e+1];@inbounds sorted[rank+1]=i;s[base+tid*E+e+1]+=Int32(1)
  end
 end
end
@kernel function general_ka!(top,sorted,experts,total,mapping,E,padE,N,cum,maxpad,maxblocks,usemap,::Val{PadE}) where {PadE}
 counts=@localmem Int32 (PadE,);scan=@localmem Int32 (1024,);tid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1)
 if bid%Int32(2)==Int32(1)
  for i in tid:Int32(1024):maxpad-Int32(1);@inbounds sorted[i+1]=N;end
 else
  start=(tid÷Int32(32))*Int32(32)
  for j in Int32(0):Int32(31);if start+j<padE;@inbounds counts[start+j+1]=Int32(0);end;end
  @synchronize()
  for i in tid:Int32(1024):N-Int32(1)
   e=@inbounds top[i+1]
   if e>=E;continue;end
   if usemap;e=@inbounds mapping[e+1];if e==Int32(-1);continue;end;end
   Atomix.@atomic :monotonic counts[e+1]+=Int32(1)
  end
  @synchronize();value=tid<E ? cld(@inbounds(counts[tid+1]),Int32(32))*Int32(32) : Int32(0);@inbounds scan[tid+1]=value;@synchronize()
  for stride in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16),Int32(32),Int32(64),Int32(128),Int32(256),Int32(512))
   idx=(tid+Int32(1))*stride*Int32(2)-Int32(1)
   if idx<Int32(1024);@inbounds scan[idx+1]+=scan[idx-stride+1];end
   @synchronize()
  end
  if tid==0;@inbounds scan[1024]=Int32(0);end
  @synchronize()
  for stride in (Int32(512),Int32(256),Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
   idx=(tid+Int32(1))*stride*Int32(2)-Int32(1)
   if idx<Int32(1024);@inbounds v=scan[idx-stride+1];scan[idx-stride+1]=scan[idx+1];scan[idx+1]+=v;end
   @synchronize()
  end
  if tid<=E;@inbounds cum[tid+1]=scan[tid+1];end
  if tid==E;@inbounds total[1]=scan[tid+1];end
  @synchronize()
  if tid<E
   for i in (@inbounds cum[tid+1]):Int32(32):(@inbounds cum[tid+2])-Int32(1);@inbounds experts[i÷Int32(32)+1]=tid;end
  end
  for i in ((@inbounds cum[E+1])÷Int32(32)+tid):Int32(1024):maxblocks-Int32(1);@inbounds experts[i+1]=Int32(0);end
 end
end
@kernel function scatter_ka!(top,sorted,cum,mapping,E,N,blocksY,usemap)
 indices=@index(Global,NTuple);tid=(Int32(indices[2])-Int32(1))*Int32(256)+Int32(indices[1])-Int32(1)
 for i in tid:Int32(256)*blocksY:N-Int32(1)
  e=@inbounds top[i+1]
  if e>=E;continue;end
  if usemap;e=@inbounds mapping[e+1];if e==Int32(-1);continue;end;end
  rank=(Atomix.@atomic :monotonic cum[e+1]+=Int32(1))-Int32(1);@inbounds sorted[rank+1]=i
 end
end
