# Source SC32-private-word in-place algorithm; barriers/Atomix backend supplied by frontend.
@inline function sc_oldadd!(a,i,v)
 pair=Atomix.@atomic :monotonic a[i]+v
 pair[1]
end
@inline function sc_scan!(mem,tx,x,sync)
 mem[tx+1]=x;offset=Int32(1);l=Int32(128)
 while l>0
  sync()
  if tx<l
   ai=offset*(2tx+1)-1;bi=offset*(2tx+2)-1;mem[bi+1]+=mem[ai+1]
  end
  offset<<=1;l>>=1
 end
 if offset<256;offset<<=1;end
 maxthread=offset>>1;d=Int32(0)
 while d<maxthread
  d+=Int32(1);offset>>=1;sync()
  if tx<d
   ai=offset*(tx+1)-1;bi=ai+(offset>>1);mem[bi+1]+=mem[ai+1]
  end
  d<<=1
 end
 sync();pos=mem[tx+1]+mem[257]-x;sync()
 if tx==255;mem[257]+=mem[256];end
 pos
end
@inline function sc_body!(data,flags,n,tasks,cut,mem,bid,tx,sync)
 my_s=cut+bid
 while my_s<tasks
  if tx==0;mem[257]=Int32(0);end
  sync()
  pos=(my_s-cut)*Int32(8192)+tx
  reg=ntuple(Val(32)) do j
   idx=pos+(j-1)*256
   idx<n ? data[idx+1] : Int32(0)
  end
  count=Int32(0)
  for j in 1:32;count+=reg[j]!=0 ? Int32(1) : Int32(0);end
  mem[tx+1]=count;sync();s=Int32(128)
  while s>0
   if tx<s;mem[tx+1]+=mem[tx+s+1];end
   sync();s>>=1
  end
  if tx==0
   mem[257]=mem[1]
   pc=Int32(0)
   while pc==0;pc=sc_oldadd!(flags,my_s+1,Int32(0));end
   sc_oldadd!(flags,my_s+2,pc+mem[257]);mem[257]=pc-Int32(1)
  end
  sync()
  for j in 1:32
   value=reg[j];target=sc_scan!(mem,tx,value!=0 ? Int32(1) : Int32(0),sync)
   if value!=0;data[target+1]=value;end
  end
  my_s+=Int32(4)
 end
 nothing
end
