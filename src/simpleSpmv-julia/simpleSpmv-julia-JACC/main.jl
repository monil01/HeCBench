using CUDA,JACC
include("../study_portable_common.jl")
@inline function dense_kernel(i,n,mat,x,y)
 i=Int32(i)-Int32(1)
 if i<n
  temp=0f0
  for j in Int32(0):n-Int32(1)
   @inbounds v=mat[i*n+j+Int32(1)]
   if v!=0f0;@inbounds temp+=v*x[j+Int32(1)];end
  end
  @inbounds y[i+Int32(1)]=temp
 end
end
@inline function csr_kernel(i,n,row,col,values,x,y)
 i=Int32(i)-Int32(1)
 if i<n
  @inbounds j=row[i+Int32(1)];stop=row[i+Int32(2)];temp=0f0
  while j<stop
   @inbounds temp+=values[j+UInt64(1)]*x[col[j+UInt64(1)]+UInt64(1)];j+=UInt64(1)
  end
  @inbounds y[i+Int32(1)]=temp
 end
end
@inline function vector_kernel(i,n,row,col,values,x,y,template,::Val{BS},::Val{T})where{BS,T}
 tid=Int32((i-1)%BS);lane=tid%Int32(T);m=Int32((i-1)÷BS)*Int32(BS÷T)+tid÷Int32(T);shared=JACC.shared(template);temp=0f0
 if m<n
  @inbounds j=row[m+Int32(1)]+UInt64(lane);stop=row[m+Int32(2)]
  while j<stop
   @inbounds temp+=values[j+UInt64(1)]*x[col[j+UInt64(1)]+UInt64(1)];j+=UInt64(T)
  end
 end
 @inbounds shared[tid+Int32(1)]=temp
 JACC.sync_workgroup()
 offset=T>>1
 while offset>0
  if lane<offset;@inbounds shared[tid+Int32(1)]+=shared[tid+Int32(offset)+Int32(1)];end
  JACC.sync_workgroup()
  offset>>=1
 end
 if lane==0&&m<n;@inbounds y[m+Int32(1)]=shared[tid+Int32(1)];end
end
const BACKEND=JACC.get_backend(:cuda)
function factory(c,bs,method,out)
 blocks=method==2 ? cld(c.n,bs÷c.tpr) : cld(c.n,bs)
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=bs,blocks=Int(blocks),shmem_size=method==2 ? bs*4 : 0,sync=false)
 if method==0
  ()->JACC.parallel_for(spec,Int(blocks)*bs,dense_kernel,c.n,c.mat,c.x,out)
 elseif method==1
  ()->JACC.parallel_for(spec,Int(blocks)*bs,csr_kernel,c.n,c.row,c.col,c.val,c.x,out)
 else
  template=CUDA.zeros(Float32,bs)
  ()->JACC.parallel_for(spec,Int(blocks)*bs,vector_kernel,c.n,c.row,c.col,c.val,c.x,out,template,Val(bs),Val(c.tpr))
 end
end
spmv_study(factory)
