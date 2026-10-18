struct ParticleParams
 cx::Float32;cy::Float32;cz::Float32;colliderRadius::Float32
 gx::Float32;gy::Float32;gz::Float32;globalDamping::Float32
 radius::Float32
 gridX::UInt32;gridY::UInt32;gridZ::UInt32;numCells::UInt32
 ox::Float32;oy::Float32;oz::Float32
 cellX::Float32;cellY::Float32;cellZ::Float32
 numBodies::UInt32;unusedMaxPerCell::UInt32
 spring::Float32;damping::Float32;shear::Float32;attraction::Float32;boundaryDamping::Float32
end
@inline vadd(a,b)=(a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
@inline function integrate_one(i,pos,vel,p,dt)
 @inbounds begin
  a=pos[i+1];b=vel[i+1]
  vx=(b[1]+p.gx*dt)*p.globalDamping;vy=(b[2]+p.gy*dt)*p.globalDamping;vz=(b[3]+p.gz*dt)*p.globalDamping
  x=a[1]+vx*dt;y=a[2]+vy*dt;z=a[3]+vz*dt;lo=-1f0+p.radius;hi=1f0-p.radius
  if x<lo;x=lo;vx*=p.boundaryDamping;end
  if x>hi;x=hi;vx*=p.boundaryDamping;end
  if y<lo;y=lo;vy*=p.boundaryDamping;end
  if y>hi;y=hi;vy*=p.boundaryDamping;end
  if z<lo;z=lo;vz*=p.boundaryDamping;end
  if z>hi;z=hi;vz*=p.boundaryDamping;end
  pos[i+1]=(x,y,z,1f0);vel[i+1]=(vx,vy,vz,0f0)
 end
 return
end
@inline gridpos(a,p)=(floor(Int32,(a[1]-p.ox)/p.cellX),floor(Int32,(a[2]-p.oy)/p.cellY),floor(Int32,(a[3]-p.oz)/p.cellZ))
@inline function gridhash(g,p)
 x=g[1]&Int32(p.gridX-UInt32(1));y=g[2]&Int32(p.gridY-UInt32(1));z=g[3]&Int32(p.gridZ-UInt32(1))
 UInt32((z*Int32(p.gridY)+y)*Int32(p.gridX)+x)
end
@inline function hash_one(i,hash,index,pos,p)
 @inbounds hash[i+1]=gridhash(gridpos(pos[i+1],p),p)
 @inbounds index[i+1]=UInt32(i)
 return
end
@inline function compare_shared!(scratch,a,b,dir)
 @inbounds if (scratch[a+1]>scratch[b+1])==dir
  scratch[a+1],scratch[b+1]=scratch[b+1],scratch[a+1]
  scratch[513+a],scratch[513+b]=scratch[513+b],scratch[513+a]
 end
 return
end
@inline function load_sort!(tid,group,hash,index,scratch)
 base=group*Int32(512)+tid
 @inbounds begin
  scratch[tid+1]=hash[base+1];scratch[tid+257]=hash[base+257]
  scratch[tid+513]=index[base+1];scratch[tid+769]=index[base+257]
 end
 return
end
@inline function store_sort!(tid,group,hash,index,scratch)
 base=group*Int32(512)+tid
 @inbounds begin
  hash[base+1]=scratch[tid+1];hash[base+257]=scratch[tid+257]
  index[base+1]=scratch[tid+513];index[base+257]=scratch[tid+769]
 end
 return
end
@inline function merge_global_one(i,hash,index,size,stride,N)
 ci=i&(N÷Int32(2)-Int32(1));dir=(ci&(size÷Int32(2)))!=0
 a=Int32(2)*i-(i&(stride-Int32(1)));b=a+stride
 @inbounds begin
  ka=hash[a+1];kb=hash[b+1];va=index[a+1];vb=index[b+1]
  if (ka>kb)==dir;ka,kb=kb,ka;va,vb=vb,va;end
  hash[a+1]=ka;hash[b+1]=kb;index[a+1]=va;index[b+1]=vb
 end
 return
end
@inline function reorder_pre(i,tid,hash,scratch,N)
 h=UInt32(0)
 @inbounds if i<N
  h=hash[i+1];scratch[tid+2]=h
  if i>0&&tid==0;scratch[1]=hash[i];end
 end
 h
end
@inline function reorder_post(i,tid,h,hash,index,pos,vel,rpos,rvel,cellStart,cellEnd,scratch,N)
 @inbounds if i<N
  if i==0
   cellStart[Int(h)+1]=UInt32(0)
  else
   previous=scratch[tid+1]
   if h!=previous;cellEnd[Int(previous)+1]=UInt32(i);cellStart[Int(h)+1]=UInt32(i);end
  end
  if i==N-Int32(1);cellEnd[Int(h)+1]=UInt32(N);end
  src=Int(index[i+1])+1;rpos[i+1]=pos[src];rvel[i+1]=vel[src]
 end
 return
end
@inline function sphere_force(a,b,va,vb,ra,rb,p)
 rx=b[1]-a[1];ry=b[2]-a[2];rz=b[3]-a[3];dist=sqrt(rx*rx+ry*ry+rz*rz);cut=ra+rb
 if dist<cut
  nx=rx/dist;ny=ry/dist;nz=rz/dist
  vx=vb[1]-va[1];vy=vb[2]-va[2];vz=vb[3]-va[3]
  dot=vx*nx+vy*ny+vz*nz;tx=vx-dot*nx;ty=vy-dot*ny;tz=vz-dot*nz;spring=-p.spring*(cut-dist)
  return (spring*nx+p.damping*vx+p.shear*tx+p.attraction*rx,spring*ny+p.damping*vy+p.shear*ty+p.attraction*ry,spring*nz+p.damping*vz+p.shear*tz+p.attraction*rz,0f0)
 end
 (0f0,0f0,0f0,0f0)
end
@inline function collide_one(i,vel,rpos,rvel,index,cellStart,cellEnd,p)
 @inbounds begin
  a=rpos[i+1];va=rvel[i+1];g=gridpos(a,p);force=(0f0,0f0,0f0,0f0)
  for z in Int32(-1):Int32(1),y in Int32(-1):Int32(1),x in Int32(-1):Int32(1)
   h=Int(gridhash((g[1]+x,g[2]+y,g[3]+z),p))+1;start=cellStart[h]
   if start==typemax(UInt32);continue;end
   stop=Int32(cellEnd[h])
   for j in Int32(start):stop-Int32(1)
    if j==i;continue;end
    force=vadd(force,sphere_force(a,rpos[j+1],va,rvel[j+1],p.radius,p.radius,p))
   end
  end
  force=vadd(force,sphere_force(a,(p.cx,p.cy,p.cz,0f0),va,(0f0,0f0,0f0,0f0),p.radius,p.colliderRadius,p))
  vel[Int(index[i+1])+1]=vadd(va,force)
 end
 return
end
