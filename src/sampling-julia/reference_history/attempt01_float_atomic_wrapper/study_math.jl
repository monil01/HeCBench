
@inline lcg_step(s::UInt64)=(UInt64(2806196910506780709)*s+UInt64(1))&UInt64(0x7fffffffffffffff)
@inline function exact_site!(tid,row,step,ncols,nbg,x,bg,obs,data)
 col=tid
 while col<ncols
  curr=@inbounds x[row*ncols+col+17]
  for donor in Int32(0):nbg-Int32(1)
   @inbounds data[(row*nbg+donor)*ncols+col+17]=curr==0f0 ? bg[donor*ncols+col+17] : obs[col+17]
  end
  col+=step
 end;nothing
end
@inline function claim_site!(tid,blk,exact,cols,x,ns,seed::UInt64)
 k=@inbounds ns[blk+17]
 if tid<k
  local_seed=seed
  while true
   local_seed=lcg_step(local_seed);idx=Int32(trunc(Float64(local_seed)/9223372036854775808.0*Float64(cols)));pos=(exact+Int32(2)*blk)*cols+idx+17
   if swap_value!(x,pos)!=1f0;break;end
  end
 end;nothing
end
@inline function sample_scatter!(tid,blk,step,exact,cols,nbg,x,bg,obs,data)
 col=tid;row=exact+Int32(2)*blk
 while col<cols
  value=@inbounds x[row*cols+col+17];@inbounds x[(row+1)*cols+col+17]=1f0-value
  for donor in Int32(0):nbg-Int32(1)
   @inbounds data[(row*nbg+donor)*cols+col+17]=value==0f0 ? bg[donor*cols+col+17] : obs[col+17]
   @inbounds data[((row+1)*nbg+donor)*cols+col+17]=value==0f0 ? obs[col+17] : bg[donor*cols+col+17]
  end
  col+=step
 end;nothing
end
