function count_kernel!(input,n::Int32,out)
 t=Int32(threadIdx().x);i=(Int32(blockIdx().x)-1)*Int32(128)+t-1
 tile=CUDA.CuStaticSharedArray(Int32,128)
 flag=Int32(0)
 @inbounds if i<n
  c=input[i+1];right=UInt8(65)<=c<=UInt8(122)
  left=i==0 ? false : UInt8(65)<=input[i]<=UInt8(122)
  flag=Int32(right&&!left)
 end
 @inbounds tile[t]=flag
 CUDA.sync_threads()
 for offset in (64,32,16,8,4,2,1)
  if t<=offset;@inbounds tile[t]+=tile[t+offset];end
  CUDA.sync_threads()
 end
 if t==1;CUDA.@atomic out[1]+=tile[1];end
 return
end
