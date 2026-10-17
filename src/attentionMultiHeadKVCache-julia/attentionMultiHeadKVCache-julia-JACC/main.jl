using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function decode_jacc!(index,q,k,v,o,n::Int32,template)
 linear=Int32(index)-Int32(1);lid=linear%Int32(512)+Int32(1);kh=linear÷Int32(512);local0=lid-Int32(1)
 tx=local0%Int32(8);ty=(local0÷Int32(8))%Int32(8);tz=local0÷Int32(64)
 # One public shared() arena: K/V converted losslessly from storage to Float32.
 # After the final tile barrier its first half is reused for merged outputs.
 arena=JACC.shared(template)
 tile=arena;outcache=arena
 md=@view arena[8193:8320]
 scratch=@view arena[8321:8832]
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
  JACC.sync_workgroup()
  scores=ntuple(i->0f0,Val(8));oldm=mv
  for j=Int32(0):Int32(7)
   score=0f0
   for i=Int32(1):Int32(8)
    @inbounds score+=qv[i]*Float32(tile[(tz*Int32(8)+j)*Int32(64)+tx*Int32(8)+i])
   end
   @inbounds scratch[lid]=score
   JACC.sync_workgroup()
   for delta in (Int32(4),Int32(2),Int32(1))
    @inbounds peer=scratch[((lid-Int32(1))⊻delta)+Int32(1)]
    JACC.sync_workgroup()
    score+=peer
    @inbounds scratch[lid]=score
    JACC.sync_workgroup()
   end
   score*=0.18033688f0 # source log2(e)/sqrt(64), roundedFloat32
   score=base+tz*Int32(8)+j<n ? score : -50000f0
   scores=Base.setindex(scores,score,Int(j)+1);mv=max(mv,score)
  end
  scale=exp2(oldm-mv);den*=scale;ov=let prev=ov, factor=scale;ntuple(i->prev[i]*factor,Val(8));end
  for j=Int32(0):Int32(7)
   weight=exp2(scores[Int(j)+1]-mv);den+=weight
   ov=let prev=ov, w=weight;ntuple(i->prev[i]+w*Float32(@inbounds tile[Int32(4096)+(tz*Int32(8)+j)*Int32(64)+tx*Int32(8)+i]),Val(8));end
  end
  JACC.sync_workgroup()
 end
 for i=Int32(1):Int32(8)
  @inbounds outcache[(tz*Int32(8)+ty)*Int32(64)+tx*Int32(8)+i]=ov[i]
 end
 if tx==Int32(0)
  @inbounds md[(tz*Int32(8)+ty)*Int32(2)+1]=mv
  @inbounds md[(tz*Int32(8)+ty)*Int32(2)+2]=den
 end
 JACC.sync_workgroup()
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
 nothing
end
const TEMPLATE=CUDA.zeros(Float32,8832)
const SPEC=JACC.launch_spec(threads=512,blocks=4,shmem_size=35328,sync=false)
function launch!(q,k,v,o,n)
 JACC.parallel_for(SPEC,2048,decode_jacc!,q,k,v,o,Int32(n),TEMPLATE)
 nothing
end
attention_study(launch!)
