# All five source kernels, shared semantic helpers; explicit CUDA device primitives.
@inline mc_field(x,z) = begin
 xf=(Float32(x)-512f0)*(1f0/512f0);zf=(Float32(z)-512f0)*(1f0/512f0)
 1f0-16f0*xf*zf*zf-4f0*(xf*xf+zf*zf+zf*zf)
end
@inline mc_coord(x)=(Float32(x)-512f0)*(1f0/512f0)
@inline mc_zero(x,v,w,iso)=((Float32(x)*(w-iso)+Float32(x+1)*(iso-v))/(w-v)-512f0)*(1f0/512f0)
@inline function mc_scan(v,tid,s,nw)
 @inbounds begin
 lane=tid%32;warp=tid÷32
 for off in (1,2,4,8,16)
  other=CUDA.shfl_up_sync(0xffffffff,v,off);if lane>=off;v+=other;end
 end
 if lane==31;s[warp+1]=v;end
 CUDA.sync_threads()
 if warp==0
  w=lane<nw ? s[lane+1] : zero(v)
  off=1
  while off<nw
   other=CUDA.shfl_up_sync(0xffffffff,w,off);if lane>=off;w+=other;end;off*=2
  end
  s[lane+1]=w
 end
 CUDA.sync_threads()
 if warp!=0;v+=s[warp];end
 return v
 end
end
@inline function mc_stage1(tid,group,out,s)
 @inbounds begin
 bx=group%68;by=(group÷68)%68;bz=group÷4624;x=bx*15+tid%16;z=bz*63
 lo=hi=mc_field(x,z)
 for k in 1:63;v=mc_field(x,z+k);lo=min(lo,v);hi=max(hi,v);end
 lane=tid%32;warp=tid÷32
 for off in (16,8,4,2,1)
  lo=min(lo,CUDA.shfl_down_sync(0xffffffff,lo,off));hi=max(hi,CUDA.shfl_down_sync(0xffffffff,hi,off))
 end
 if lane==0;s[warp+1]=lo;s[warp+9]=hi;end
 CUDA.sync_threads()
 if warp==0
  lo=lane<8 ? s[lane+1] : Inf32;hi=lane<8 ? s[lane+9] : -Inf32
  for off in (4,2,1)
   lo=min(lo,CUDA.shfl_down_sync(0xffffffff,lo,off));hi=max(hi,CUDA.shfl_down_sync(0xffffffff,hi,off))
  end
  if lane==0;out[2group+1]=lo;out[2group+2]=hi;end
 end
 end;return
end
@inline function mc_stage2(tid,group,mm,ids,count,s,iso)
 @inbounds begin
 id=128group+tid;test=UInt32(mm[2id+1]<=iso<=mm[2id+2]);v=mc_scan(test,tid,s,4)
 if tid==127 && v!=0;s[32]=CUDA.atomic_add!(pointer(count,1),v);end
 CUDA.sync_threads()
 if test!=0;ids[Int(v+s[32])]=UInt32(id);end
 end;return
end
@inline function mc_stage3(thread,group,ids,mm)
 @inbounds begin
 tid=thread%16;vo=thread÷16;parent=Int32(ids[group+1]);x=(parent%68)*15+(vo%5)*3+(tid&3);z=(parent÷4624)*63
 v=mc_field(x,z);lo=hi=v;idx=2*(vo+225group)
 for blockz in 0:8
  for k in 1:7;v=mc_field(x,z+k);lo=min(lo,v);hi=max(hi,v);end;z+=7
  for off in (8,4,2,1)
   lo=min(lo,CUDA.shfl_down_sync(0xffffffff,lo,off));hi=max(hi,CUDA.shfl_down_sync(0xffffffff,hi,off))
  end
  if tid==0;mm[idx+1]=lo;mm[idx+2]=hi;idx+=50;end
  lo=hi=v
 end
 end;return
end
@inline function mc_stage4(tid,group,mm,ids1,ids2,count,n1,s,iso)
 @inbounds begin
 id=1024group+tid;p=id÷225;test=p<n1 ? UInt32(mm[2id+1]<=iso<=mm[2id+2]) : UInt32(0);v=mc_scan(test,tid,s,32)
 if tid==1023;s[32]=CUDA.atomic_add!(pointer(count,1),v);end
 CUDA.sync_threads()
 if test!=0
  parent=Int32(ids1[p+1]);sub=id%225;x=parent%68*5+sub%5;y=(parent÷68)%68*5+(sub÷5)%5;z=parent÷4624*9+sub÷25
  ids2[Int(v+s[32])]=UInt32(x+340*(y+340z))
 end
 end;return
end
@inline mc_vi(x,y,z)=x+4*(y+4z)+1
@inline mc_fi(x,y,z)=x+5*(y+5z)+65
@inline function mc_stage5(tid,group,ids,dist,tri,edges,counts,checksum,coords,raw,iso)
 @inbounds begin
 vi=reinterpret(UInt16,raw);values=reinterpret(Float32,raw)
 tx=tid&3;ty=(tid÷4)&3;tz=tid÷16;block=Int32(ids[group+1]);x=(block%340)*3+tx;y=((block÷340)%340)*3+ty;z=(block÷115600)*7+tz
 eds=UInt32(7);v=mc_field(x,z);values[mc_fi(tx,ty,tz)]=v
 if tx==3
  eds&=UInt32(6);values[mc_fi(4,ty,tz)]=mc_field(x+1,z)
  if ty==3;values[mc_fi(4,4,tz)]=mc_field(x+1,z);end
 end
 if ty==3
  eds&=UInt32(5);values[mc_fi(tx,4,tz)]=mc_field(x,z)
  if tz==7;values[mc_fi(tx,4,8)]=mc_field(x,z+1);end
 end
 if tz==7
  eds&=UInt32(3);values[mc_fi(tx,ty,8)]=mc_field(x,z+1)
  if tx==3;values[mc_fi(4,ty,8)]=mc_field(x+1,z+1);end
 end
 eds<<=13;CUDA.sync_threads();case=Int32(0)
 # Unused all-far corner is read only by the no-owner lane; avoid undefined read.
 if eds!=0
  for (bit,dx,dy,dz) in ((0,0,0,0),(1,1,0,0),(2,1,1,0),(3,0,1,0),(4,0,0,1),(5,1,0,1),(6,1,1,1),(7,0,1,1))
   if values[mc_fi(tx+dx,ty+dy,tz+dz)]<iso;case|=Int32(1)<<bit;end
  end
 end
 de=eds!=0 ? UInt32(dist[case+1]) : UInt32(0);nt=eds==0xe000 ? de&UInt32(7) : UInt32(0);de&=eds;nv=UInt32(count_ones(de))
 # Shared subviews preserve UInt32 storage and source warp scan structure.
 sv=typeof(raw)(pointer(raw,290),(32,));st=typeof(raw)(pointer(raw,322),(32,))
 vs=mc_scan(nv,tid,sv,4);ts=mc_scan(nt,tid,st,4)
 if tid==127;sv[32]=CUDA.atomic_add!(pointer(counts,1),vs);st[32]=CUDA.atomic_add!(pointer(counts,2),ts);end
 CUDA.sync_threads();base=sv[32];vi[tid+1]=UInt16((vs-nv)|de);CUDA.sync_threads()
 for it in Int32(0):Int32(nt)-Int32(1),corner in Int32(0):Int32(2)
  edge=Int32(tri[16case+3it+corner+1]);e=4edge;dx=Int32(edges[e+1]);dy=Int32(edges[e+2]);dz=Int32(edges[e+3]);w=Int32(edges[e+4]);vv=UInt32(vi[mc_vi(tx+dx,ty+dy,tz+dz)]);localid=UInt32(count_ones(vv>>(16-w)))+(vv&UInt32(0x1fff));CUDA.atomic_add!(pointer(checksum,1),UInt64(base+localid))
 end
 cx=cy=cz=zp=0f0
 if de&0x8000!=0;zp=mc_zero(x,v,values[mc_fi(tx+1,ty,tz)],iso);cy=mc_coord(y);cz=mc_coord(z);end
 if de&0x4000!=0;cx=mc_coord(x);zp+=mc_zero(y,v,values[mc_fi(tx,ty+1,tz)],iso);cz+=mc_coord(z);end
 if de&0x2000!=0;cx+=mc_coord(x);cy+=mc_coord(y);zp+=mc_zero(z,v,values[mc_fi(tx,ty,tz+1)],iso);end
 CUDA.atomic_add!(pointer(coords,1),cx);CUDA.atomic_add!(pointer(coords,2),cy);CUDA.atomic_add!(pointer(coords,3),cz);CUDA.atomic_add!(pointer(coords,4),zp)
 end;return
end
