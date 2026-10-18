@inline rgb(p::UInt32)=(Int32(p&0xff),Int32((p>>8)&0xff),Int32((p>>16)&0xff))
@inline function putrgb!(cache,t,p)
 c=rgb(p);@inbounds begin cache[3*t+1]=c[1];cache[3*t+2]=c[2];cache[3*t+3]=c[3];end;nothing
end
@inline function getrgb(cache,t)
 @inbounds (cache[3*t+1],cache[3*t+2],cache[3*t+3])
end
@inline function forward_costs(right,left,up)
 pr=abs(right[1]-left[1]);pg=abs(right[2]-left[2]);pb=abs(right[3]-left[3]);u=pr+pg+pb
 l=u+abs(up[1]-left[1])+abs(up[2]-left[2])+abs(up[3]-left[3]);r=u+abs(up[1]-right[1])+abs(up[2]-right[2])+abs(up[3]-right[3]);(Int16(l),Int16(u),Int16(r))
end
@inline function cost_load!(cache,pixels,t,bx,by,W,H,cw,approx)
 tx=t%Int32(32);ty=t÷Int32(32);stride=approx ? Int32(28) : Int32(30);halo=approx ? Int32(2) : Int32(1);row=by*Int32(7)+ty-Int32(1);col=bx*stride+tx-halo
 p=zero(UInt32);@inbounds if 0<=row<H&&0<=col<cw;p=pixels[row*W+col+1];end;putrgb!(cache,t,p);nothing
end
@inline function cost_store!(cache,L,U,R,t,bx,by,W,H,cw)
 tx=t%Int32(32);ty=t÷Int32(32);row=by*Int32(7)+ty-Int32(1);col=bx*Int32(30)+tx-Int32(1)
 if 0<=row<H&&0<=col<cw&&ty!=0&&tx!=0&&tx!=31
  c=forward_costs(getrgb(cache,t+1),getrgb(cache,t-1),getrgb(cache,t-32));i=row*W+col+1;@inbounds L[i]=c[1];@inbounds U[i]=c[2];@inbounds R[i]=c[3]
 end;nothing
end
@inline function dp_initialize!(cache,L,U,R,M,t)
 @inbounds z=min(Int32(L[t+1]),min(Int32(U[t+1]),Int32(R[t+1])));@inbounds cache[t+1]=z;@inbounds M[t+1]=z;nothing
end
@inline function dp_row!(cache,L,U,R,M,t,W,cw,row,base)
 ix=row*W+t+1;@inbounds begin
  l=t>0 ? cache[base+t]+Int32(L[ix]) : typemax(Int32);u=cache[base+t+1]+Int32(U[ix]);r=t<cw-1 ? cache[base+t+2]+Int32(R[ix]) : typemax(Int32);z=min(l,min(u,r));M[ix]=z;cache[cw-base+t+1]=z
 end;nothing
end
@inline function reduce_load!(cache,M,indices,t,cw,base)
 value=typemax(Int32);idx=Int32(0);col=t
 @inbounds for k in 1:8
  if col<cw;newidx=indices[col+1];newval=M[base+newidx+1];if newval<value;idx=newidx;value=newval;end;end;col+=Int32(128)
 end
 @inbounds cache[t+1]=value;@inbounds cache[128+t+1]=idx;nothing
end
@inline function reduce_step!(cache,t,stride)
 @inbounds if t<stride
  a=cache[t+stride+1];b=cache[t+1];ai=cache[128+t+stride+1];bi=cache[128+t+1];if a<b||(a==b&&ai<bi);cache[t+1]=a;cache[128+t+1]=ai;end
 end;nothing
end
@inline function backtrack!(M,indices,seam,W,H,cw)
 @inbounds begin
  ix=indices[1];seam[H]=ix
  for row in H-Int32(2):-Int32(1):Int32(0)
   mid=ix;base=row*W;if mid!=0&&M[base+mid]<M[base+ix+1];ix=mid-Int32(1);end
   # Preserve original logical-right-boundary defect and omitted transition costs.
   if mid!=cw&&M[base+mid+2]<M[base+ix+1];ix=mid+Int32(1);end;seam[row+1]=ix
  end
 end;nothing
end
@inline function remove_lane!(p,out,seam,col,row,W,H,cw)
 @inbounds if row<H&&col<cw-1;i=row*W+col+1;out[i]=p[i+Int32(col>=seam[row+1])];end;nothing
end
@inline function update_lane!(p,L,U,R,LS,US,RS,seam,col,row,W,H,cw)
 @inbounds if row<H&&col<cw-1
  s=seam[row+1];i=row*W+col+1
  if s-2<=col<=s+1
   zero_rgb=(Int32(0),Int32(0),Int32(0));right=col==cw-2 ? zero_rgb : rgb(p[i+1]);left=col==0 ? zero_rgb : rgb(p[i-1]);up=row==0 ? zero_rgb : rgb(p[i-W]);c=forward_costs(right,left,up);LS[i]=c[1];US[i]=c[2];RS[i]=c[3]
  else;j=i+Int32(col>s+1);LS[i]=L[j];US[i]=U[j];RS[i]=R[j];end
 end;nothing
end
@inline function approx_costs!(cache,t,bx,by,W,H,cw)
 tx=t%Int32(32);ty=t÷Int32(32);row=by*Int32(7)+ty-Int32(1);col=bx*Int32(28)+tx-Int32(2)
 if 0<=row<H&&0<=col<cw&&ty>0
  # Source halo0/31 computes unused undefined locals; define those unobservable values.
  z=(Int32(0),Int32(0),Int32(0));right=tx<31 ? getrgb(cache,t+1) : z;left=tx>0 ? getrgb(cache,t-1) : z;c=forward_costs(right,left,getrgb(cache,t-32));shorts=reinterpret(Int16,cache)
  @inbounds shorts[1536+t+1]=c[1];@inbounds shorts[1792+t+1]=c[2];@inbounds shorts[2048+t+1]=c[3]
 end;nothing
end
@inline function approx_select!(cache,index,offset,M,t,bx,by,W,H,cw)
 tx=t%Int32(32);ty=t÷Int32(32);row=by*Int32(7)+ty-Int32(1);col=bx*Int32(28)+tx-Int32(2)
 if 0<=row<H-1&&0<=col<cw&&1<tx<30&&ty!=7
  shorts=reinterpret(Int16,cache);value=typemax(Int32);dest=Int32(0);i=row*W+col+1
  @inbounds begin
   if col>0;value=Int32(shorts[2048+t+32]);dest=(row+1)*W+col-1;end
   c=Int32(shorts[1792+t+32+1]);if c<value;value=c;dest=(row+1)*W+col;end
   if col<cw-1;c=Int32(shorts[1536+t+32+2]);if c<value;value=c;dest=(row+1)*W+col+1;end;end
   index[i]=dest;offset[i]=dest;M[i]=value
  end
 end;nothing
end
@inline function approx_sum_lane!(offset,M,col,by,W,H,cw,step)
 row=by*2*step
 @inbounds if row+step<H-1&&col<cw;i=row*W+col+1;dest=offset[i]+1;M[i]+=M[dest];offset[i]=offset[dest];end;nothing
end
@inline function approx_backtrack!(index,indices,seam,W,H)
 @inbounds begin
  ix=indices[1];for row in Int32(0):H-Int32(1);seam[row+1]=ix-row*W;if row<H-1;ix=index[ix+1];end;end
 end;nothing
end
