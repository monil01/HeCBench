@inline function reduce_body!(tid,s,limit)
 d=Int32(128)
 while d>0
  if tid<Int32(128)
   if tid<d&&tid+d<limit;@inbounds s[tid+1]=min(s[tid+1],s[tid+d+1]);end
  else
   idx=tid-Int32(128)
   if idx<d&&idx+d<limit;@inbounds s[256+idx+1]=max(s[256+idx+1],s[256+idx+d+1]);end
  end
  block_barrier();d>>=1
 end
 return
end
@inline function scan_body!(tid,bid,grid,lo,hi,input,num,s)
 @inbounds n=Int32(num[1]);blocks=cld(n,Int32(256));small=Int32(0);large=Int32(0);last=Int32(0);block=bid
 while block<blocks
  offset=block*Int32(256);ending=min(n-offset,Int32(256));last=max(last,ending)
  if tid<ending
   @inbounds val=input[offset+tid+1]
   if block==bid;small=val;large=val;else;small=min(small,val);large=max(large,val);end
  end
  block+=grid
 end
 @inbounds s[tid+1]=small;s[256+tid+1]=large
 block_barrier();reduce_body!(tid,s,last)
 if tid==0;@inbounds lo[bid+1]=s[1];hi[bid+1]=s[257];end
 return
end
@inline function finalize_body!(tid,lo,hi,bits,minvalue,num,s)
 @inbounds n=min(cld(Int32(num[1]),Int32(256)),Int32(4096))
 if tid<n
  @inbounds small=lo[tid+1];large=hi[tid+1];i=tid+Int32(256)
  while i<n;@inbounds small=min(lo[i+1],small);large=max(hi[i+1],large);i+=Int32(256);end
  @inbounds s[tid+1]=small;s[256+tid+1]=large
 end
 block_barrier();reduce_body!(tid,s,min(Int32(256),n))
 if tid==0
  @inbounds minvalue[1]=s[1];range=UInt32(s[257]-s[1]);bits[1]=UInt8(32-leading_zeros(range))
 end
 return
end
@inline function pack_body!(tid,bid,grid,bits,minvalue,out,input,num,s)
 @inbounds n=Int32(num[1]);nb=Int32(bits[1]);minimum=minvalue[1]
 blocks=cld(n,Int32(256));block=bid
 while block<blocks
  outputidx=tid+block*Int32(256);startbit=outputidx*Int32(32);endbit=startbit+Int32(32);start=min(startbit÷nb,n);ending=min(cld(endbit,nb),n)
  blockstartbit=block*Int32(8192);blockendbit=(block+Int32(1))*Int32(8192);bufferstart=min((blockstartbit÷nb)÷Int32(256)*Int32(256),n);bufferend=min(cld(cld(blockendbit,nb),Int32(256))*Int32(256),n);value=UInt32(0)
  while bufferstart<bufferend
   block_barrier();index=bufferstart+tid
   if index<n;@inbounds s[tid+1]=UInt32(input[index+1]-minimum);end
   block_barrier();first=max(start,bufferstart);last=min(ending,bufferstart+Int32(256))
   for index in first:last-Int32(1)
    @inbounds shifted=s[index-bufferstart+1];off=index*nb-startbit
    value|=off>0 ? shifted<<off : shifted>>(-off)
   end
   bufferstart+=Int32(256)
  end
  if start<n;@inbounds out[outputidx+1]=value;end
  block+=grid
 end
 return
end
