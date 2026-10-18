const DEP_STEPS=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
@inline function dep_color(rgb,i)
 r=rgb[3i+1];g=rgb[3i+2];b=rgb[3i+3]
 y=0.299f0*r+0.587f0*g+0.114f0*b
 u=0.713f0*(r-y)+0.5f0;v=0.564f0*(b-y)+0.5f0
 yy=unsafe_trunc(UInt32,clamp(y,0f0,1f0)*255f0);uu=unsafe_trunc(UInt32,clamp(u,0f0,1f0)*255f0);vv=unsafe_trunc(UInt32,clamp(v,0f0,1f0)*255f0)
 return (UInt32(255)<<24)|(vv<<16)|(uu<<8)|yy
end
@inline function dep_connected(a,b)
 y=Int32(a&UInt32(255));u=Int32((a>>8)&UInt32(255));v=Int32((a>>16)&UInt32(255))
 yy=Int32(b&UInt32(255));uu=Int32((b>>8)&UInt32(255));vv=Int32((b>>16)&UInt32(255))
 return !(abs(y-yy)>48||abs(u-uu)>7||abs(v-vv)>6)
end
@inline function dep_check!(rgb,out,center,w,h)
 row=center÷w;col=center%w;yc=dep_color(rgb,center);con=UInt32(0)
 for bit in 0:7
  dr,dc=DEP_STEPS[bit+1];rr=row+Int32(dr);cc=col+Int32(dc);inside=0<=rr<h&&0<=cc<w
  neighbor=inside ? rr*w+cc : center;yn=dep_color(rgb,neighbor)
  con+=UInt32(inside&&dep_connected(yc,yn))<<bit
 end
 out[center+Int32(1)]=((yc>>16)&UInt32(255))<<24|((yc>>8)&UInt32(255))<<16|(yc&UInt32(255))<<8|con
 return
end
@inline function dep_bitcount(v)
 c=Int32(0)
 while v!=0;v&=v-UInt32(1);c+=Int32(1);end
 return c
end
@inline function dep_walk(id,w,h,row,col,incoming,score)
 curve=id[row*w+col+Int32(1)]&UInt32(255)
 while dep_bitcount(curve)==2&&score<w*h
  edge=curve-incoming
  if edge==1;row-=Int32(1);col-=Int32(1)
  elseif edge==2;row-=Int32(1)
  elseif edge==4;row-=Int32(1);col+=Int32(1)
  elseif edge==8;col+=Int32(1)
  elseif edge==16;row+=Int32(1);col+=Int32(1)
  elseif edge==32;row+=Int32(1)
  elseif edge==64;row+=Int32(1);col-=Int32(1)
  elseif edge==128;col-=Int32(1)
  end
  incoming=edge>8 ? edge>>4 : edge<<4
  curve=id[row*w+col+Int32(1)]&UInt32(255);score+=Int32(1)
 end
 return score
end
@inline function dep_eliminate!(id,out,center,w,h)
 row=center÷w;col=center%w;value=UInt32(0);own=id[center+Int32(1)]
 if row<h-1&&col<w-1
  right=id[center+Int32(2)];br=id[center+w+Int32(2)]
  value=(own&UInt32(8))>>3|(((br&UInt32(2))>>1)<<1)|(((br&UInt32(128))>>7)<<2)|(((own&UInt32(32))>>5)<<3)
  if own&UInt32(16)!=0&&right&UInt32(64)!=0
   if own&UInt32(40)!=0&&right&UInt32(160)!=0
    out[center+Int32(1)]=(own&UInt32(0xffffff00))|value;return
   end
   wl=(own&UInt32(255))==16 ? Int32(5) : Int32(0);wr=(right&UInt32(255))==64 ? Int32(5) : Int32(0)
   sl=Int32(0);sr=Int32(0)
   for rr in max(row-Int32(3),Int32(0)):min(row+Int32(4),h-Int32(1)),cc in max(col-Int32(3),Int32(0)):min(col+Int32(4),w-Int32(1))
    j=rr*w+cc
    if j!=center&&j!=center+1
     other=id[j+Int32(1)]>>8;sl+=Int32(dep_connected(own>>8,other));sr+=Int32(dep_connected(right>>8,other))
    end
   end
   wr+=max(sl-sr,Int32(0));wl+=max(sr-sl,Int32(0))
   sl=dep_walk(id,w,h,row,col,UInt32(16),Int32(1));sl=dep_walk(id,w,h,row+Int32(1),col+Int32(1),UInt32(1),sl)
   sr=dep_walk(id,w,h,row,col+Int32(1),UInt32(64),Int32(1));sr=dep_walk(id,w,h,row+Int32(1),col,UInt32(4),sr)
   wl+=max(sl-sr,Int32(0));wr+=max(sr-sl,Int32(0))
   if wl>wr;value|=UInt32(16);out[center+Int32(1)]=(own&UInt32(0xffffff00))|value;return
   elseif wr>wl;value|=UInt32(32);out[center+Int32(1)]=(own&UInt32(0xffffff00))|value;return
   end
  end
  value|=(((own&UInt32(16))>>4)<<4)|(((right&UInt32(64))>>6)<<5)
 end
 out[center+Int32(1)]=(own&UInt32(0xffffff00))|value
 return
end
