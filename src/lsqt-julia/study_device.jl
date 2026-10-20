# GPUQT equations translated to Julia by Codex,2026-10-19.
# Original copyright2017 Zheyong Fan,Ville Vierimaa,Ari Harju;GPLv3-or-later.
# Pure arithmetic bodies; each target owns dispatch/index/shared storage.
@inline function body_zero!(i,n,re,im)
 if i<n;re[i+1]=0.0;im[i+1]=0.0;end;nothing
end
@inline function body_copy!(i,n,ar,ai,br,bi)
 if i<n;br[i+1]=ar[i+1];bi[i+1]=ai[i+1];end;nothing
end
@inline function body_add!(i,n,ar,ai,br,bi)
 if i<n;br[i+1]+=ar[i+1];bi[i+1]+=ai[i+1];end;nothing
end
@inline function body_sz!(i,n,ar,ai,br,bi)
 if i<n
  if iseven(i);br[i+1]=ar[i+1];bi[i+1]=ai[i+1];else;br[i+1]=-ar[i+1];bi[i+1]=-ai[i+1];end
 end;nothing
end
@inline function warp_reduce!(shared,t,backend)
 v=shared[t+1]
 for offset in (32,16,8,4,2)
  v+=shared[t+offset+1];warp_sync(backend);shared[t+1]=v;warp_sync(backend)
 end
 v+=shared[t+2];warp_sync(backend);shared[t+1]=v;nothing
end
@inline function reduce_pair!(tid,sr,si,backend)
 block_sync(backend)
 if tid<128;sr[tid+1]+=sr[tid+129];si[tid+1]+=si[tid+129];end;block_sync(backend)
 if tid<64;sr[tid+1]+=sr[tid+65];si[tid+1]+=si[tid+65];end;block_sync(backend)
 if tid<32;warp_reduce!(sr,tid,backend);warp_reduce!(si,tid,backend);end;nothing
end
@inline function body_ip1!(tid,block,n,ar,ai,br,bi,orr,oi,offset,sr,si,backend)
 i=block*256+tid;sr[tid+1]=0.0;si[tid+1]=0.0
 if i<n
  a=ar[i+1];b=ai[i+1];c=br[i+1];d=bi[i+1];sr[tid+1]=muladd(a,c,b*d);si[tid+1]=muladd(b,c,-a*d)
 end
 reduce_pair!(tid,sr,si,backend)
 if tid==0;orr[block+offset+1]=sr[1];oi[block+offset+1]=si[1];end;nothing
end
@inline function body_ip2!(tid,block,n,ar,ai,orr,oi,sr,si,backend)
 blocks=cld(n,Int32(256));patches=cld(blocks,Int32(256));sr[tid+1]=0.0;si[tid+1]=0.0
 for patch=0:patches-1
  k=tid+patch*256
  if k<blocks;at=block*blocks+k+1;sr[tid+1]+=ar[at];si[tid+1]+=ai[at];end
 end
 reduce_pair!(tid,sr,si,backend)
 if tid==0;orr[block+1]=sr[1];oi[block+1]=si[1];end;nothing
end
@inline function H_element(i,n,emax,nn,nl,pot,hr,hi,ar,ai)
 tr=pot[i+1]*ar[i+1];ti=pot[i+1]*ai[i+1]
 for k=0:nn[i+1]-1
  at=k*n+i+1;j=nl[at]+1;a=hr[at];b=hi[at];c=ar[j];d=ai[j]
  tr+=muladd(a,c,-b*d);ti+=muladd(a,d,b*c)
 end;tr/emax,ti/emax
end
@inline function body_H!(i,n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi)
 if i<n;tr,ti=H_element(i,n,emax,nn,nl,pot,hr,hi,ar,ai);orr[i+1]=tr;oi[i+1]=ti;end;nothing
end
@inline function body_commutator!(i,n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 if i<n
  tr=0.0;ti=0.0
  for k=0:nn[i+1]-1
   at=k*n+i+1;j=nl[at]+1;a=hr[at];b=hi[at];c=ar[j];d=ai[j];x=dx[at]
   tr-=muladd(a,c,-b*d)*x;ti-=muladd(a,d,b*c)*x
  end
  orr[i+1]=tr/emax;oi[i+1]=ti/emax
 end;nothing
end
@inline function body_current!(i,n,nn,nl,hr,hi,dx,ar,ai,orr,oi)
 if i<n
  tr=0.0;ti=0.0
  for k=0:nn[i+1]-1
   at=k*n+i+1;j=nl[at]+1;a=hr[at];b=hi[at];c=ar[j];d=ai[j]
   tr+=muladd(a,c,-b*d)*dx[at];ti+=muladd(a,d,b*c)*dx[at]
  end
  orr[i+1]=ti;oi[i+1]=-tr
 end;nothing
end
@inline function body_cheb01!(i,n,r0,i0,r1,i1,orr,oi,b0,b1,direction)
 if i<n;b=b1*direction;orr[i+1]=muladd(b0,r0[i+1],b*i1[i+1]);oi[i+1]=muladd(b0,i0[i+1],-b*r1[i+1]);end;nothing
end
@inline function accumulate!(i,orr,oi,tr,ti,b,label)
 if label==1;orr[i+1]+=b*tr;oi[i+1]+=b*ti
 elseif label==2;orr[i+1]-=b*tr;oi[i+1]-=b*ti
 elseif label==3;orr[i+1]+=b*ti;oi[i+1]-=b*tr
 elseif label==4;orr[i+1]-=b*ti;oi[i+1]+=b*tr;end;nothing
end
@inline function body_cheb2!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label)
 if i<n
  tr,ti=H_element(i,n,emax,nn,nl,pot,hr,hi,r1,i1);tr=muladd(2.0,tr,-r0[i+1]);ti=muladd(2.0,ti,-i0[i+1]);accumulate!(i,orr,oi,tr,ti,b,label);r2[i+1]=tr;i2[i+1]=ti
 end;nothing
end
@inline function body_cheb1x!(i,n,r1x,i1x,orr,oi,b)
 if i<n;orr[i+1]=b*i1x[i+1];oi[i+1]=-b*r1x[i+1];end;nothing
end
@inline function body_cheb2x!(i,n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label)
 if i<n
  tr=pot[i+1]*r1[i+1];ti=pot[i+1]*i1[i+1];txr=pot[i+1]*r1x[i+1];txi=pot[i+1]*i1x[i+1]
  for k=0:nn[i+1]-1
   at=k*n+i+1;j=nl[at]+1;a=hr[at];bb=hi[at];c=r1[j];d=i1[j];cr=r1x[j];ci=i1x[j]
   tr+=muladd(a,c,-bb*d);ti+=muladd(a,d,bb*c);txr+=muladd(a,cr,-bb*ci);txi+=muladd(a,ci,bb*cr)
   txr-=muladd(a,c,-bb*d)*dx[at];txi-=muladd(a,d,bb*c)*dx[at]
  end
  tr=muladd(2.0,tr/emax,-r0[i+1]);ti=muladd(2.0,ti/emax,-i0[i+1]);r2[i+1]=tr;i2[i+1]=ti
  txr=muladd(2.0,txr/emax,-r0x[i+1]);txi=muladd(2.0,txi/emax,-i0x[i+1]);r2x[i+1]=txr;i2x[i+1]=txi;accumulate!(i,orr,oi,txr,txi,b,label)
 end;nothing
end
@inline function body_poly!(i,n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2)
 if i<n
  tr,ti=H_element(i,n,emax,nn,nl,pot,hr,hi,r1,i1);r2[i+1]=muladd(2.0,tr,-r0[i+1]);i2[i+1]=muladd(2.0,ti,-i0[i+1])
 end;nothing
end
