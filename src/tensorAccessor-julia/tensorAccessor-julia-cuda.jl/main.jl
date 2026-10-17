using CUDA
function raw_study!(nr,nc,r,m,v)
 i=Int64((blockIdx().x-Int32(1))*blockDim().x+threadIdx().x)
 if i<=nr
  val=0f0
  @inbounds for j in Int64(1):nc;val=muladd(m[(i-1)*nc+j],v[j],val);end
  @inbounds r[i]=val
 end
 return
end
function packed_study!(r,m,v)
 i=Int64((blockIdx().x-Int32(1))*blockDim().x+threadIdx().x)
 if i<=size(r,1)
  val=0f0
  for j in Int64(1):size(m,2);val=muladd(m[i][j],v[j],val);end
  r[i]=val
 end
 return
end
