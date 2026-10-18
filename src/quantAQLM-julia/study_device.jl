
const QF=Int32(9)
@inline function q_load!(sh,B,bvec,base,tx,nt,k)
 i=tx
 @inbounds while i<256
  if base+i<k÷8
   dst=(QF*(i÷8)+i%8)*8
   src=bvec+(base+i)*8
   for j=0:7;sh[dst+j+1]=B[src+j+1];end
  end
  i+=nt
 end
 nothing
end
@inline function q_accum(sh,A,book,row,lane,tile,k,m,acc)
 @inbounds if row<m && lane+tile*32<k÷64
  part=row<m÷3 ? Int32(0) : (row<2*(m÷3) ? Int32(1) : Int32(2))
  for enc=Int32(0):Int32(7)
   code=Int32(A[row*(k÷8)+(lane+tile*32)*8+enc+1])
   off=(part*65536+code)*8
   sb=(QF*lane+enc)*8
   lo=Float16(0);hi=Float16(0)
   for j=Int32(0):Int32(3)
    lo=fma(book[off+2j+1],sh[sb+2j+1],lo)
    hi=fma(book[off+2j+2],sh[sb+2j+2],hi)
   end
   acc+=Float32(lo)+Float32(hi)
  end
 end
 acc
end
@inline function q_reduce_store!(C,cvec,row,lane,m,res)
 if row<m
  for off in (16,8,4,2,1);res+=CUDA.shfl_down_sync(typemax(UInt32),res,off);end
  @inbounds if lane==0;C[cvec+row+1]=Float16(res);end
 end
 nothing
end
