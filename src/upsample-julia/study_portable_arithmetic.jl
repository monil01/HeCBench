@inline function tile!(x,out,H::UInt64,W::UInt64,bc::UInt64,i::UInt64,j::UInt64,::Val{BACK}) where BACK
 base=bc*4H*W+4i*W+2j
 flat=bc*H*W+i*W+j
 @inbounds if BACK
  out[Int(flat)+1]=x[Int(base)+1]+x[Int(base)+2]+x[Int(base+2W)+1]+x[Int(base+2W)+2]
 else
  v=x[Int(flat)+1];out[Int(base)+1]=v;out[Int(base)+2]=v;out[Int(base+2W)+1]=v;out[Int(base+2W)+2]=v
 end
 return
end
