using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel function dense_kernel(n,mat,x,y)
 i=Int32(@index(Global,Linear))-Int32(1)
 if i<n
  temp=0f0
  for j in Int32(0):n-Int32(1)
   @inbounds v=mat[i*n+j+Int32(1)]
   if v!=0f0;@inbounds temp+=v*x[j+Int32(1)];end
  end
  @inbounds y[i+Int32(1)]=temp
 end
end
@kernel function csr_kernel(n,row,col,values,x,y)
 i=Int32(@index(Global,Linear))-Int32(1)
 if i<n
  @inbounds j=row[i+Int32(1)];stop=row[i+Int32(2)];temp=0f0
  while j<stop
   @inbounds temp+=values[j+UInt64(1)]*x[col[j+UInt64(1)]+UInt64(1)];j+=UInt64(1)
  end
  @inbounds y[i+Int32(1)]=temp
 end
end
@kernel unsafe_indices=true function vector_kernel(n,row,col,values,x,y,::Val{BS},::Val{T})where{BS,T}
 tid=Int32(@index(Local,Linear))-Int32(1);lane=tid%Int32(T);m=(Int32(@index(Group,Linear))-Int32(1))*Int32(BS÷T)+tid÷Int32(T);shared=@localmem Float32 (BS,);temp=0f0
 if m<n
  @inbounds j=row[m+Int32(1)]+UInt64(lane);stop=row[m+Int32(2)]
  while j<stop
   @inbounds temp+=values[j+UInt64(1)]*x[col[j+UInt64(1)]+UInt64(1)];j+=UInt64(T)
  end
 end
 @inbounds shared[tid+Int32(1)]=temp
 @synchronize
 offset=T>>1
 while offset>0
  if lane<offset;@inbounds shared[tid+Int32(1)]+=shared[tid+Int32(offset)+Int32(1)];end
  @synchronize
  offset>>=1
 end
 if lane==0&&m<n;@inbounds y[m+Int32(1)]=shared[tid+Int32(1)];end
end
function factory(c,bs,method,out)
 if method==0
  k=dense_kernel(CUDABackend(),bs);()->k(c.n,c.mat,c.x,out;ndrange=cld(c.n,bs)*bs)
 elseif method==1
  k=csr_kernel(CUDABackend(),bs);()->k(c.n,c.row,c.col,c.val,c.x,out;ndrange=cld(c.n,bs)*bs)
 else
  k=vector_kernel(CUDABackend(),bs);()->k(c.n,c.row,c.col,c.val,c.x,out,Val(bs),Val(c.tpr);ndrange=cld(c.n,bs÷c.tpr)*bs)
 end
end
spmv_study(factory)
