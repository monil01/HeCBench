using CUDA,KernelAbstractions
@kernel function local_sort_ka!(keys,tmp,startbit)
 local_index=@index(Local,Linear);tid=Int32(local_index-1);group_index=@index(Group,Linear);bid=Int32(group_index-1);s=@localmem UInt32 (512,);base=(bid*Int32(128)+tid)*Int32(4)
 @inbounds begin;k0=keys[base+1];k1=keys[base+2];k2=keys[base+3];k3=keys[base+4];end
 for shift in startbit:startbit+Int32(3)
  p0=UInt32(((k0>>shift)&UInt32(1))==0);p1=UInt32(((k1>>shift)&UInt32(1))==0);p2=UInt32(((k2>>shift)&UInt32(1))==0);p3=UInt32(((k3>>shift)&UInt32(1))==0);total=p0+p1+p2+p3;x=total;@inbounds s[tid+1]=x
  @synchronize
  for offset in (1,2,4,8,16,32,64)
   neighbor=tid>=offset ? (@inbounds s[tid-offset+1]) : UInt32(0)
   @synchronize;x+=neighbor;@inbounds s[tid+1]=x;@synchronize
  end
  alltrue=@inbounds s[128];e=x-total;e0=e;e1=e+p0;e2=e1+p1;e3=e2+p2;i=UInt32(tid)*UInt32(4)
  r0=p0!=0 ? e0 : alltrue+i-e0;r1=p1!=0 ? e1 : alltrue+i+UInt32(1)-e1;r2=p2!=0 ? e2 : alltrue+i+UInt32(2)-e2;r3=p3!=0 ? e3 : alltrue+i+UInt32(3)-e3
  @synchronize
  @inbounds begin;s[(r0&UInt32(3))*UInt32(128)+(r0>>2)+1]=k0;s[(r1&UInt32(3))*UInt32(128)+(r1>>2)+1]=k1;s[(r2&UInt32(3))*UInt32(128)+(r2>>2)+1]=k2;s[(r3&UInt32(3))*UInt32(128)+(r3>>2)+1]=k3;end
  @synchronize
  @inbounds begin;k0=s[tid+1];k1=s[tid+129];k2=s[tid+257];k3=s[tid+385];end
  @synchronize
 end
 @inbounds begin;tmp[base+1]=k0;tmp[base+2]=k1;tmp[base+3]=k2;tmp[base+4]=k3;end
end
@kernel function offsets_ka!(tmp,counts,blocks,startbit,nb)
 local_index=@index(Local,Linear);tid=Int32(local_index-1);group_index=@index(Group,Linear);bid=Int32(group_index-1);s=@localmem UInt32 (272,);base=bid*Int32(256)+tid*Int32(2)
 @inbounds begin;s[2tid+1]=(tmp[base+1]>>startbit)&UInt32(15);s[2tid+2]=(tmp[base+2]>>startbit)&UInt32(15);end
 if tid<16;@inbounds s[257+tid]=UInt32(0);end
 @synchronize
 if tid>0&&(@inbounds s[tid+1]!=s[tid]);@inbounds s[257+Int32(s[tid+1])]=UInt32(tid);end
 if @inbounds s[tid+129]!=s[tid+128];@inbounds s[257+Int32(s[tid+129])]=UInt32(tid+128);end
 @synchronize
 if tid<16;@inbounds blocks[bid*Int32(16)+tid+1]=s[257+tid];end
 @synchronize
 if tid>0&&(@inbounds s[tid+1]!=s[tid]);@inbounds s[257+Int32(s[tid])]=UInt32(tid)-s[257+Int32(s[tid])];end
 if @inbounds s[tid+129]!=s[tid+128];@inbounds s[257+Int32(s[tid+128])]=UInt32(tid+128)-s[257+Int32(s[tid+128])];end
 if tid==127;@inbounds s[257+Int32(s[256])]=UInt32(256)-s[257+Int32(s[256])];end
 @synchronize
 if tid<16;@inbounds counts[tid*nb+bid+1]=s[257+tid];end
end
@kernel function scan1_ka!(dest,src)
 local_index=@index(Local,Linear);tid=Int32(local_index-1);group_index=@index(Group,Linear);bid=Int32(group_index-1);s=@localmem UInt32 (512,);base=(bid*Int32(256)+tid)*Int32(4)
 @inbounds begin;v0=src[base+1];v1=src[base+2];v2=src[base+3];v3=src[base+4];end
 total=v0+v1+v2+v3;pos=tid+Int32(256);@inbounds begin;s[tid+1]=UInt32(0);s[pos+1]=total;end
 value=total
 for offset in (1,2,4,8,16,32,64,128)
  @synchronize;value=(@inbounds s[pos+1]+s[pos-offset+1]);@synchronize;@inbounds s[pos+1]=value
 end
 prefix=value-total
 @inbounds begin;dest[base+1]=prefix;dest[base+2]=prefix+v0;dest[base+3]=prefix+v0+v1;dest[base+4]=prefix+v0+v1+v2;end
end
@kernel function scan2_ka!(buffer,dest,src,bn)
 local_index=@index(Local,Linear);tid=Int32(local_index-1);s=@localmem UInt32 (512,);data=UInt32(0)
 if tid<bn;index=(tid+Int32(1))*Int32(1024);data=(@inbounds dest[index]+src[index]);end
 pos=Int32(2)*tid-(tid&(bn-Int32(1)));@inbounds s[pos+1]=UInt32(0);pos+=bn;@inbounds s[pos+1]=data;value=data;offset=Int32(1)
 while offset<bn
  @synchronize;value=(@inbounds s[pos+1]+s[pos-offset+1]);@synchronize;@inbounds s[pos+1]=value;offset*=Int32(2)
 end
 if tid<bn;@inbounds buffer[tid+1]=value-data;end
end
@kernel function uniform_ka!(dest,buffer)
 local_index=@index(Local,Linear);tid=Int32(local_index-1);group_index=@index(Group,Linear);bid=Int32(group_index-1);s=@localmem UInt32 (1,)
 if tid==0;@inbounds s[1]=buffer[bid+1];end
 @synchronize
 value=@inbounds s[1];base=(bid*Int32(256)+tid)*Int32(4)
 @inbounds begin;dest[base+1]+=value;dest[base+2]+=value;dest[base+3]+=value;dest[base+4]+=value;end
end
@kernel function reorder_ka!(keys,tmp,blocks,scan,startbit,nb,n)
 local_index=@index(Local,Linear);tid=Int32(local_index-1);group_index=@index(Group,Linear);bid=Int32(group_index-1);s=@localmem UInt32 (288,);base=bid*Int32(256)+tid*Int32(2)
 @inbounds begin;s[2tid+1]=tmp[base+1];s[2tid+2]=tmp[base+2];end
 if tid<16;@inbounds begin;s[257+tid]=scan[tid*nb+bid+1];s[273+tid]=blocks[bid*Int32(16)+tid+1];end;end
 @synchronize
 for pos in (tid,tid+Int32(128))
  key=@inbounds s[pos+1];digit=Int32((key>>startbit)&UInt32(15));offset=(@inbounds s[257+digit])+UInt32(pos)-(@inbounds s[273+digit])
  if offset<UInt32(n);@inbounds keys[offset+1]=key;end
 end
end
