function count_jacc!(g,input,n::Int32,out,template)
 t=Int32(mod(g-1,128)+1);i=Int32(g-1)
 tile=JACC.shared(template)
 flag=Int32(0)
 @inbounds if i<n
  c=input[i+1];right=UInt8(65)<=c<=UInt8(122)
  left=i==0 ? false : UInt8(65)<=input[i]<=UInt8(122)
  flag=Int32(right&&!left)
 end
 @inbounds tile[t]=flag
 JACC.sync_workgroup()
 for offset in (64,32,16,8,4,2,1)
  if t<=offset;@inbounds tile[t]+=tile[t+offset];end
  JACC.sync_workgroup()
 end
 if t==1;Atomix.@atomic out[1]+=tile[1];end
 return
end
