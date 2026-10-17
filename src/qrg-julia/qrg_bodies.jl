@inline function qrg_body!(idx::UInt32,stride::UInt32,output,table,seed::UInt32,total::UInt32,n::UInt32)
 while idx<total
  dim=idx÷n;pos=idx-dim*n;result=UInt32(0);data=seed+pos
  @inbounds for bit=UInt32(0):UInt32(QRNG_RESOLUTION-1)
   if (data&UInt32(1))!=UInt32(0);result⊻=table[Int(bit+dim*UInt32(QRNG_RESOLUTION)+UInt32(1))];end
   data>>=UInt32(1)
  end
  @inbounds output[Int(dim*n+pos+UInt32(1))]=Float32(result+UInt32(1))*INT_SCALE_F32
  idx+=stride
 end
 nothing
end
@inline function icnd_body!(pos::UInt32,stride::UInt32,output,total::UInt32,distance::UInt32)
 while pos<total
  @inbounds output[Int(pos+UInt32(1))]=moro_inv_cnd_gpu((pos+UInt32(1))*distance)
  pos+=stride
 end
 nothing
end
