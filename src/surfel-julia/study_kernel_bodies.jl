@inline function candidate_depth(s,base,rayx,rayy,rayz,best)
 @inbounds px=s[base+1];py=s[base+2];pz=s[base+3];nx=s[base+4];ny=s[base+5];nz=s[base+6];radius2=s[base+7]
 pn=px*nx+py*ny+pz*nz;rn=rayx*nx+rayy*ny+rayz*nz;alpha=pn/rn;dx=rayx*alpha-px;dy=rayy*alpha-py;dz=rayz*alpha-pz;depth=rayz*alpha;distance2=dx*dx+dy*dy+dz*dz
 return distance2<radius2&&best>depth ? depth : best
end
@inline function base_body!(x,y,s,N,f,w,h,d)
 if x<w&&y<h
  rayx=Float32(x)-Float32(w-1)*.5f0;rayy=Float32(y)-Float32(h-1)*.5f0;best=1f20
  for i in Int32(0):N-Int32(1);best=candidate_depth(s,i*Int32(7),rayx,rayy,f,best);end
  @inbounds d[y*w+x+1]=best>100f0 ? 0f0 : best
 end
 return
end
@inline function tile_body!(x,y,tid,s,N,f,w,h,d,shared)
 if x>=w||y>=h;return;end
 rayx=Float32(x)-Float32(w-1)*.5f0;rayy=Float32(y)-Float32(h-1)*.5f0;best=1f20;base=Int32(0)
 while base<N
  if base+tid<N
   for k in Int32(0):Int32(6);@inbounds shared[tid*Int32(7)+k+1]=s[(base+tid)*Int32(7)+k+1];end
  end
  block_barrier();count=min(Int32(256),N-base)
  for i in Int32(0):count-Int32(1);best=candidate_depth(shared,i*Int32(7),rayx,rayy,f,best);end
  block_barrier();base+=Int32(256)
 end
 @inbounds d[y*w+x+1]=best>100f0 ? 0f0 : best
 return
end
