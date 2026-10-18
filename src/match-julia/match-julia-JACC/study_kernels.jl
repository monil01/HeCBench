function match1!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,128));ty=Int32(j-1);bx=Int32(div(i-1,128))
 @inbounds begin
p=tx+128*bx;best=0f0;ix=Int32(-1)
for q in 0:127
 v=0f0
 for d in 0:127;v=muladd(a[p*128+d+1],b[q*128+d+1],v);end
 if v>best;best=v;ix=Int32(q);end
end
out[p+1]=best;idx[p+1]=ix
end
return
end
function match2!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,16));ty=Int32(j-1);bx=Int32(div(i-1,16))
 @inbounds begin
s=JACC.shared(template);bp=16*bx;li=tx+16*ty;best=0f0;ix=Int32(-1)
for d in tx:16:127;s[ty*128+d+1]=a[(bp+ty)*128+d+1];end
JACC.sync_workgroup()
for bp2 in 0:16:127
for d in tx:16:127;s[2048+ty*128+d+1]=b[(bp2+ty)*128+d+1];end
JACC.sync_workgroup()
v=0f0
for d in 0:127;v=muladd(s[tx*128+d+1],s[2048+ty*128+d+1],v);end
s[4096+li+1]=v
JACC.sync_workgroup()
if ty==0
for j in 0:15
v=s[4096+16*j+tx+1];if v>best;best=v;ix=Int32(bp2+j);end
end
end
JACC.sync_workgroup()
end
if ty==0;out[bp+tx+1]=best;idx[bp+tx+1]=ix;end
end
return
end
function match3!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,16));ty=Int32(j-1);bx=Int32(div(i-1,16))
 @inbounds begin
s=JACC.shared(template);bp=16*bx;li=tx+16*ty;best=0f0;ix=Int32(-1)
for d in tx:16:127;s[ty*129+d+1]=a[(bp+ty)*128+d+1];end
JACC.sync_workgroup()
for bp2 in 0:16:127
for d in tx:16:127;s[2064+ty*128+d+1]=b[(bp2+ty)*128+d+1];end
JACC.sync_workgroup()
v=0f0
for d in 0:127;v=muladd(s[tx*129+d+1],s[2064+ty*128+d+1],v);end
s[4112+li+1]=v
JACC.sync_workgroup()
if ty==0
for j in 0:15
v=s[4112+16*j+tx+1];if v>best;best=v;ix=Int32(bp2+j);end
end
end
JACC.sync_workgroup()
end
if ty==0;out[bp+tx+1]=best;idx[bp+tx+1]=ix;end
end
return
end
function match4!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,16));ty=Int32(j-1);bx=Int32(div(i-1,16))
 @inbounds begin
s=JACC.shared(template);bp=16*bx;li=tx+16*ty;best=0f0;ix=Int32(-1)
for d in tx:16:31,c in 0:3;s[ty*132+4*d+c+1]=a[(bp+ty)*128+4*d+c+1];end
JACC.sync_workgroup()
for bp2 in 0:16:127
for d in tx:16:31,c in 0:3;s[2112+ty*128+4*d+c+1]=b[(bp2+ty)*128+4*d+c+1];end
JACC.sync_workgroup()
v=0f0
for d in 0:127;v=muladd(s[tx*132+d+1],s[2112+ty*128+d+1],v);end
s[4160+li+1]=v
JACC.sync_workgroup()
if ty==0
for j in 0:15
v=s[4160+16*j+tx+1];if v>best;best=v;ix=Int32(bp2+j);end
end
end
JACC.sync_workgroup()
end
if ty==0;out[bp+tx+1]=best;idx[bp+tx+1]=ix;end
end
return
end
function match5!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,16));ty=Int32(j-1);bx=Int32(div(i-1,16))
 @inbounds begin
s=JACC.shared(template);bp=16*bx;li=tx+16*ty;best=0f0;ix=Int32(-1)
for d in tx:16:31,c in 0:3;s[ty*132+4*d+c+1]=a[(bp+ty)*128+4*d+c+1];end
JACC.sync_workgroup()
for bp2 in 0:16:127
for d in tx:16:31,c in 0:3;s[2112+ty*128+4*d+c+1]=b[(bp2+ty)*128+4*d+c+1];end
JACC.sync_workgroup()
if ty<4
vals=(0f0,0f0,0f0,0f0)
for d in 0:31,c in 0:3
x=s[tx*132+4*d+c+1]
for q in 0:3;vals=Base.setindex(vals,muladd(x,s[2112+(4*ty+q)*128+4*d+c+1],vals[q+1]),q+1);end
end
for q in 0:3;s[4160+tx+16*(4*ty+q)+1]=vals[q+1];end
end
JACC.sync_workgroup()
if ty==0
for q in 0:15;v=s[4160+16*q+tx+1];if v>best;best=v;ix=Int32(bp2+q);end;end
end
JACC.sync_workgroup()
end
if ty==0;out[bp+tx+1]=best;idx[bp+tx+1]=ix;end
end
return
end
function match6!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,16));ty=Int32(j-1);bx=Int32(div(i-1,16))
 @inbounds begin
s=JACC.shared(template);bp=16*bx;li=tx+16*ty;best=0f0;ix=Int32(-1)
for d in tx:16:31,c in 0:3;s[ty*132+4*d+c+1]=a[(bp+ty)*128+4*d+c+1];end
for bp2 in 0:16:127
for d in tx:16:31,c in 0:3;s[2112+ty*128+4*d+c+1]=b[(bp2+ty)*128+4*d+c+1];end
JACC.sync_workgroup()
if ty<4
vals=(0f0,0f0,0f0,0f0)
for d in 0:31,c in 0:3
x=s[tx*132+4*d+c+1]
for q in 0:3;vals=Base.setindex(vals,muladd(x,s[2112+(4*ty+q)*128+4*d+c+1],vals[q+1]),q+1);end
end
for q in 0:3;v=vals[q+1];if v>best;best=v;ix=Int32(bp2+4*ty+q);end;end
end
JACC.sync_workgroup()
end
if ty<4
s[ty*16+tx+1]=best;s[64+ty*16+tx+1]=reinterpret(Float32,ix)
end
JACC.sync_workgroup()
if ty==0
best=s[tx+1];ix=reinterpret(Int32,s[64+tx+1])
for q in 0:3
v=s[q*16+tx+1];if v>best;best=v;ix=reinterpret(Int32,s[64+q*16+tx+1]);end
end
out[bp+tx+1]=best;idx[bp+tx+1]=ix
end
end
return
end
function match7!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,32));ty=Int32(j-1);bx=Int32(div(i-1,32))
 @inbounds begin
s=JACC.shared(template);bp=32*bx;li=tx+32*ty;best=0f0;ix=Int32(-1)
for j in ty:8:31,c in 0:3;s[j*128+4*mod(tx+j,32)+c+1]=a[(bp+j)*128+4*tx+c+1];end
for bp2 in 0:32:127
for j in ty:8:31,c in 0:3;s[4096+j*128+4*tx+c+1]=b[(bp2+j)*128+4*tx+c+1];end
JACC.sync_workgroup()
vals=(0f0,0f0,0f0,0f0)
for d in 0:31,c in 0:3
x=s[tx*128+4*mod(d+tx,32)+c+1]
for q in 0:3;vals=Base.setindex(vals,muladd(x,s[4096+(4*ty+q)*128+4*d+c+1],vals[q+1]),q+1);end
end
for q in 0:3;v=vals[q+1];if v>best;best=v;ix=Int32(bp2+4*ty+q);end;end
JACC.sync_workgroup()
end
s[ty*32+tx+1]=best;s[256+ty*32+tx+1]=reinterpret(Float32,ix)
JACC.sync_workgroup()
if ty==0
best=s[tx+1];ix=reinterpret(Int32,s[256+tx+1])
for q in 0:7
v=s[q*32+tx+1];if v>best;best=v;ix=reinterpret(Int32,s[256+q*32+tx+1]);end
end
out[bp+tx+1]=best;idx[bp+tx+1]=ix
end
end
return
end
function match8!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,32));ty=Int32(j-1);bx=Int32(div(i-1,32))
 @inbounds begin
s=JACC.shared(template);bp=32*bx;li=tx+32*ty;best=0f0;ix=Int32(-1)
for j in ty:8:31,c in 0:3;s[j*128+4*mod(tx+j,32)+c+1]=a[(bp+j)*128+4*tx+c+1];end
ixx=mod(li,16);iyy=div(li,16);best2=0f0;ix2=Int32(-1)
for bp2 in 0:32:127
for j in ty:8:31,c in 0:3;s[4096+j*128+4*tx+c+1]=b[(bp2+j)*128+4*tx+c+1];end
JACC.sync_workgroup()
if li<128
v1=(0f0,0f0,0f0,0f0);v2=v1
for d in 0:31,c in 0:3
x=s[ixx*128+4*mod(d+ixx,32)+c+1];y=s[(16+ixx)*128+4*mod(d+16+ixx,32)+c+1]
for q in 0:3;z=s[4096+(4*iyy+q)*128+4*d+c+1];v1=Base.setindex(v1,muladd(x,z,v1[q+1]),q+1);v2=Base.setindex(v2,muladd(y,z,v2[q+1]),q+1);end
end
for q in 0:3
if v1[q+1]>best;best=v1[q+1];ix=Int32(bp2+4*iyy+q);end
if v2[q+1]>best2;best2=v2[q+1];ix2=Int32(bp2+4*iyy+q);end
end
end
JACC.sync_workgroup()
end
if li<128
s[iyy*32+ixx+1]=best;s[iyy*32+16+ixx+1]=best2;s[256+iyy*32+ixx+1]=reinterpret(Float32,ix);s[256+iyy*32+16+ixx+1]=reinterpret(Float32,ix2)
end
JACC.sync_workgroup()
if ty==0
best=s[tx+1];ix=reinterpret(Int32,s[256+tx+1])
for q in 0:7
v=s[q*32+tx+1];if v>best;best=v;ix=reinterpret(Int32,s[256+q*32+tx+1]);end
end
out[bp+tx+1]=best;idx[bp+tx+1]=ix
end
end
return
end
function match9!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,32));ty=Int32(j-1);bx=Int32(div(i-1,32))
 @inbounds begin
s=JACC.shared(template);bp=32*bx;li=tx+32*ty;best=0f0;ix=Int32(-1)
for j in ty:4:31,c in 0:3;s[j*128+4*mod(tx+j,32)+c+1]=a[(bp+j)*128+4*tx+c+1];end
ixx=mod(li,16);iyy=div(li,16);best2=0f0;ix2=Int32(-1)
for bp2 in 0:32:127
for j in ty:4:31,c in 0:3;s[4096+j*128+4*tx+c+1]=b[(bp2+j)*128+4*tx+c+1];end
JACC.sync_workgroup()
v1=(0f0,0f0,0f0,0f0);v2=v1
for d in 0:31,c in 0:3
x=s[ixx*128+4*mod(d+ixx,32)+c+1];y=s[(16+ixx)*128+4*mod(d+16+ixx,32)+c+1]
for q in 0:3;z=s[4096+(4*iyy+q)*128+4*d+c+1];v1=Base.setindex(v1,muladd(x,z,v1[q+1]),q+1);v2=Base.setindex(v2,muladd(y,z,v2[q+1]),q+1);end
end
for q in 0:3
if v1[q+1]>best;best=v1[q+1];ix=Int32(bp2+4*iyy+q);end
if v2[q+1]>best2;best2=v2[q+1];ix2=Int32(bp2+4*iyy+q);end
end
JACC.sync_workgroup()
end
if li<128
s[iyy*32+ixx+1]=best;s[iyy*32+16+ixx+1]=best2;s[256+iyy*32+ixx+1]=reinterpret(Float32,ix);s[256+iyy*32+16+ixx+1]=reinterpret(Float32,ix2)
end
JACC.sync_workgroup()
if ty==0
best=s[tx+1];ix=reinterpret(Int32,s[256+tx+1])
for q in 0:7
v=s[q*32+tx+1];if v>best;best=v;ix=reinterpret(Int32,s[256+q*32+tx+1]);end
end
out[bp+tx+1]=best;idx[bp+tx+1]=ix
end
end
return
end
function match10!(i,j,a,b,out,idx,template)
 tx=Int32(mod(i-1,32));ty=Int32(j-1);bx=Int32(div(i-1,32))
 @inbounds begin
s=JACC.shared(template);bp=32*bx;li=tx+32*ty;best=0f0;ix=Int32(-1)
for j in ty:8:31,c in 0:3;s[j*128+4*mod(tx+j,32)+c+1]=a[(bp+j)*128+4*tx+c+1];end
ixx=mod(li,16);iyy=div(li,16);best2=0f0;ix2=Int32(-1)
for bp2 in 0:32:127
v1=(0f0,0f0,0f0,0f0);v2=v1
dd=mod(li,8);jj=div(li,8)
for c in 0:3;s[4096+jj*32+4*dd+c+1]=b[(bp2+jj)*128+4*dd+c+1];end
JACC.sync_workgroup()
for dp in 0:8:31
pref=(dp<24 ? b[(bp2+jj)*128+4*(dp+dd+8)+1] : 0f0,dp<24 ? b[(bp2+jj)*128+4*(dp+dd+8)+2] : 0f0,dp<24 ? b[(bp2+jj)*128+4*(dp+dd+8)+3] : 0f0,dp<24 ? b[(bp2+jj)*128+4*(dp+dd+8)+4] : 0f0)
if li<128
for d in 0:7,c in 0:3
x=s[ixx*128+4*mod(dp+d+ixx,32)+c+1];y=s[(16+ixx)*128+4*mod(dp+d+16+ixx,32)+c+1]
for q in 0:3;z=s[4096+(4*iyy+q)*32+4*d+c+1];v1=Base.setindex(v1,muladd(x,z,v1[q+1]),q+1);v2=Base.setindex(v2,muladd(y,z,v2[q+1]),q+1);end
end
end
JACC.sync_workgroup()
if dp<24
for c in 0:3;s[4096+jj*32+4*dd+c+1]=pref[c+1];end
JACC.sync_workgroup()
end
end
if li<128
for q in 0:3
if v1[q+1]>best;best=v1[q+1];ix=Int32(bp2+4*iyy+q);end
if v2[q+1]>best2;best2=v2[q+1];ix2=Int32(bp2+4*iyy+q);end
end
end
JACC.sync_workgroup()
end
if li<128
s[iyy*32+ixx+1]=best;s[iyy*32+16+ixx+1]=best2;s[256+iyy*32+ixx+1]=reinterpret(Float32,ix);s[256+iyy*32+16+ixx+1]=reinterpret(Float32,ix2)
end
JACC.sync_workgroup()
if ty==0
best=s[tx+1];ix=reinterpret(Int32,s[256+tx+1])
for q in 0:7
v=s[q*32+tx+1];if v>best;best=v;ix=reinterpret(Int32,s[256+q*32+tx+1]);end
end
out[bp+tx+1]=best;idx[bp+tx+1]=ix
end
end
return
end
