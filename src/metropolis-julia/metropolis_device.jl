# Corrected field-to-actual-spin mapping; original native body preserved separately.
@inline ci(x,y,z)=z*1024+y*32+x+1
@inline si(x,y,z)=(z+1)*180+(y+1)*18+x+2
@inline wrap32(x)=x<0 ? x+32 : x>=32 ? x-32 : x
@inline function pcg_draw(st::UInt64,inc::UInt64)
 old=st;st=old*UInt64(6364136223846793005)+inc
 xo=UInt32((((old>>18)⊻old)>>27)&UInt64(0xffffffff));rot=Int(old>>59);word=(xo>>rot)|(xo<<((-rot)&31))
 st,Float32(word)*2.3283064f-10
end
@inline function tile_load!(ss,S,H,A,I,tx,ty,tz,bx,by,bz,alt)
 x=bx*16+tx;oy=(2*by+((bx+bz+alt)&1))*8;y=oy+2*ty;z=bz*4+tz;sy=2*ty
 @inbounds begin
  ss[si(tx,sy,tz)]=S[ci(x,y,z)];ss[si(tx,sy+1,tz)]=S[ci(x,y+1,z)]
  if ty==0;ss[si(tx,-1,tz)]=S[ci(x,wrap32(oy-1),z)];end
  if ty==3;ss[si(tx,8,tz)]=S[ci(x,wrap32(oy+8),z)];end
  if tx==0
   ss[si(-1,sy,tz)]=S[ci(wrap32(x-1),y,z)];ss[si(-1,sy+1,tz)]=S[ci(wrap32(x-1),y+1,z)]
  end
  if tx==15
   ss[si(16,sy,tz)]=S[ci(wrap32(x+1),y,z)];ss[si(16,sy+1,tz)]=S[ci(wrap32(x+1),y+1,z)]
  end
  if tz==0
   ss[si(tx,sy,-1)]=S[ci(x,y,wrap32(z-1))];ss[si(tx,sy+1,-1)]=S[ci(x,y+1,wrap32(z-1))]
  end
  if tz==3
   ss[si(tx,sy,4)]=S[ci(x,y,wrap32(z+1))];ss[si(tx,sy+1,4)]=S[ci(x,y+1,wrap32(z+1))]
  end
  wy=sy+((tx+tz)&1);byy=sy+((tx+tz+1)&1);tid=z*256+(by*4+ty)*32+x+1
  h1=H[ci(x,oy+wy,z)];h2=H[ci(x,oy+byy,z)]
  return (tx,sy,tz,x,y,z,tid,wy,byy,h1,h2),A[tid],I[tid]
 end
end
@inline function metropolis_phase!(ss,c,h,B,st,inc,::Val{P}) where P
 tx,sy,tz,x,y,z,tid,wy,byy,h1,h2=c;yy=P==0 ? wy : byy;field=P==0 ? h1 : h2
 @inbounds begin
  spin=ss[si(tx,yy,tz)];nn=ss[si(tx-1,yy,tz)]+ss[si(tx+1,yy,tz)]+ss[si(tx,yy-1,tz)]+ss[si(tx,yy+1,tz)]+ss[si(tx,yy,tz-1)]+ss[si(tx,yy,tz+1)]
  dh=Float32(spin)*muladd(h,Float32(field),Float32(nn));st,u=pcg_draw(st,inc);flip=signbit(dh-1f-11)|signbit(u-exp(dh*B));ss[si(tx,yy,tz)]=spin*(Int32(1)-Int32(2)*Int32(flip))
 end
 st
end
@inline function tile_store!(ss,S,A,I,c,st,inc)
 tx,sy,tz,x,y,z,tid=c
 @inbounds begin
  S[ci(x,y,z)]=ss[si(tx,sy,tz)];S[ci(x,y+1,z)]=ss[si(tx,sy+1,tz)];A[tid]=st;I[tid]=inc
 end
 nothing
end
