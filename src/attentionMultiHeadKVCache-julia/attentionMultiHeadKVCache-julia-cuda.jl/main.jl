using CUDA
function decode_cuda!(q,k,v,o,n::Int32)
 tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);tz=Int32(threadIdx().z)-Int32(1);kh=Int32(blockIdx().y)-Int32(1)
 tile=CuDynamicSharedArray(eltype(q),8192)
 outcache=CuDynamicSharedArray(Float32,4096)
 md=CuDynamicSharedArray(Float32,128,16384)

 qh=kh*Int32(8)+ty
 qv=ntuple(i->Float32(@inbounds q[qh*Int32(64)+tx*Int32(8)+i]),Val(8))
 mv=-50000f0;den=1f0;ov=ntuple(i->0f0,Val(8))
 for base=Int32(0):Int32(64):n-Int32(1)
  pos=base+tz*Int32(8)+ty
  for i=Int32(1):Int32(8)
   slot=(tz*Int32(8)+ty)*Int32(64)+tx*Int32(8)+i
   idx=pos*Int32(256)+kh*Int32(64)+tx*Int32(8)+i
   @inbounds tile[slot]=pos<n ? k[idx] : zero(eltype(k))
   @inbounds tile[slot+Int32(4096)]=pos<n ? v[idx] : zero(eltype(v))
  end
  CUDA.sync_threads()
  scores=ntuple(i->0f0,Val(8));oldm=mv
  for j=Int32(0):Int32(7)
   score=0f0
   for i=Int32(1):Int32(8)
    @inbounds score+=qv[i]*Float32(tile[(tz*Int32(8)+j)*Int32(64)+tx*Int32(8)+i])
   end
   for delta in (Int32(4),Int32(2),Int32(1));score+=CUDA.shfl_xor_sync(UInt32(0xffffffff),score,delta);end
   score*=0.18033688f0 # source log2(e)/sqrt(64), roundedFloat32
   score=base+tz*Int32(8)+j<n ? score : -50000f0
   scores=Base.setindex(scores,score,Int(j)+1);mv=max(mv,score)
  end
  scale=exp2(oldm-mv);den*=scale;ov=let prev=ov, factor=scale;ntuple(i->prev[i]*factor,Val(8));end
  for j=Int32(0):Int32(7)
   weight=exp2(scores[Int(j)+1]-mv);den+=weight
   ov=let prev=ov, w=weight;ntuple(i->prev[i]+w*Float32(@inbounds tile[Int32(4096)+(tz*Int32(8)+j)*Int32(64)+tx*Int32(8)+i]),Val(8));end
  end
  CUDA.sync_threads()
 end
 for i=Int32(1):Int32(8)
  @inbounds outcache[(tz*Int32(8)+ty)*Int32(64)+tx*Int32(8)+i]=ov[i]
 end
 if tx==Int32(0)
  @inbounds md[(tz*Int32(8)+ty)*Int32(2)+1]=mv
  @inbounds md[(tz*Int32(8)+ty)*Int32(2)+2]=den
 end
 CUDA.sync_threads()
 mv=-50000f0;den=1f0;ov=ntuple(i->0f0,Val(8))
 for z=Int32(0):Int32(7)
  @inbounds zm=md[(z*Int32(8)+ty)*Int32(2)+1]
  @inbounds zd=md[(z*Int32(8)+ty)*Int32(2)+2]
  newm=max(mv,zm);a=exp2(mv-newm);b=exp2(zm-newm)
  den=den*a+zd*b
  ov=let prev=ov, sa=a, sb=b;ntuple(i->prev[i]*sa+Float32(@inbounds outcache[(z*Int32(8)+ty)*Int32(64)+tx*Int32(8)+i])*sb,Val(8));end;mv=newm
 end
 # Source has identical duplicate stores by everyz; oneowner removeswriteconflict.
 if tz==Int32(0)
  for i=Int32(1):Int32(8)
   @inbounds o[qh*Int32(64)+tx*Int32(8)+i]=eltype(o)(ov[i]/den)
  end
 end
 return
end
