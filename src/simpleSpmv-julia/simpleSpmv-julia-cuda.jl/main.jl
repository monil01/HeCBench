using CUDA
include("../study_common.jl")
include("../reference_kernels.jl")
function vector_kernel(n,row,col,values,x,y,::Val{BS})where{BS}
 tid=Int32(threadIdx().x)-Int32(1);lane=tid%Int32(BS);m=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x÷BS)+tid÷Int32(BS)
 if m<n
  temp=0f0;j=@inbounds row[m+Int32(1)]+UInt64(lane);stop=@inbounds row[m+Int32(2)]
  while j<stop
   @inbounds temp+=values[j+UInt64(1)]*x[col[j+UInt64(1)]+UInt64(1)]
   j+=UInt64(BS)
  end
  offset=BS>>1
  while offset>0
   temp+=CUDA.shfl_down_sync(UInt32(0xffffffff),temp,offset,BS);offset>>=1
  end
  if lane==0;@inbounds y[m+Int32(1)]=temp;end
 end
 return
end
function factory(c,bs,method,out)
 if method==0
  ()->(@cuda threads=bs blocks=cld(c.n,bs) mv_dense_kernel!(c.n,c.mat,c.x,out))
 elseif method==1
  ()->(@cuda threads=bs blocks=cld(c.n,bs) mv_csr_kernel!(c.n,c.row,c.col,c.val,c.x,out))
 else
  ()->(@cuda threads=bs blocks=cld(c.n,bs÷c.tpr) vector_kernel(c.n,c.row,c.col,c.val,c.x,out,Val(c.tpr)))
 end
end
spmv_study(factory)
