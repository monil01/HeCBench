using KernelAbstractions
import Atomix
@kernel function count_ka!(input,n::Int32,out)
 lc=@index(Local,Linear)
 g=@index(Global,Linear)
 t=Int32(lc);i=Int32(g-1)
 tile=(@localmem Int32 (128,))
 flag=Int32(0)
 @inbounds if i<n
  c=input[i+1];right=UInt8(65)<=c<=UInt8(122)
  left=i==0 ? false : UInt8(65)<=input[i]<=UInt8(122)
  flag=Int32(right&&!left)
 end
 @inbounds tile[t]=flag
 @synchronize
 for offset in (64,32,16,8,4,2,1)
  if t<=offset;@inbounds tile[t]+=tile[t+offset];end
  @synchronize
 end
 if t==1;Atomix.@atomic out[1]+=tile[1];end
end
