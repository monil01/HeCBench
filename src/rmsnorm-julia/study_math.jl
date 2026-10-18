using CUDA
@inline function rms_load(x,xv,offset,::Val{1})
 @inbounds return (x[offset+1],)
end
@inline function rms_load(x,xv,offset,::Val{4})
 @inbounds a=xv[(offset>>2)+1]
 return (a[1].value,a[2].value,a[3].value,a[4].value)
end
@inline function rms_store!(o,ov,offset,a,::Val{1})
 @inbounds o[offset+1]=a[1]
 return
end
@inline function rms_store!(o,ov,offset,a,::Val{4})
 @inbounds ov[(offset>>2)+1]=(VecElement(a[1]),VecElement(a[2]),VecElement(a[3]),VecElement(a[4]))
 return
end
@inline function rms_warp_sum(value)
 for offset in (16,8,4,2,1)
  other=CUDA.shfl_xor_sync(UInt32(0xffffffff),value,Int32(offset));value=other+value
 end
 return value
end
@inline function rms_partial(x,xv,row,tid,H,::Val{BT},::Val{U}) where {BT,U}
 warp=(tid-Int32(1))>>5;lane=(tid-Int32(1))&Int32(31);start=warp*Int32(32U)+lane*Int32(U);sum=0f0
 for offset in start:Int32(BT*U):H-Int32(1)
  a=rms_load(x,xv,(row-Int32(1))*H+offset,Val(U))
  for j in 1:U;sum=muladd(a[j],a[j],sum);end
 end
 return rms_warp_sum(sum)
end
@inline function rms_finish!(x,xv,g,gv,o,ov,row,tid,H,total,::Val{BT},::Val{U}) where {BT,U}
 warp=(tid-Int32(1))>>5;lane=(tid-Int32(1))&Int32(31);start=warp*Int32(32U)+lane*Int32(U);factor=CUDA.rsqrt(total/Float32(H)+1f-5)
 for offset in start:Int32(BT*U):H-Int32(1)
  pos=(row-Int32(1))*H+offset;a=rms_load(x,xv,pos,Val(U));b=rms_load(g,gv,offset,Val(U))
  values=let aa=a,bb=b,f=factor;ntuple(j->aa[j]*f*bb[j],Val(U));end
  rms_store!(o,ov,pos,values,Val(U))
 end
 return
end
