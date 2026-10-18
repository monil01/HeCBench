struct Point3
 x::Float32
 y::Float32
 z::Float32
end
const Normal4=NTuple{4,VecElement{Float32}}
@inline n4(x,y,z,w)=(VecElement(Float32(x)),VecElement(Float32(y)),VecElement(Float32(z)),VecElement(Float32(w)))
@inline sub3(a,b)=(a.x-b.x,a.y-b.y,a.z-b.z)
@inline function normal_site!(points,out,idx,W,H)
 if idx<=W*H
  @inbounds begin
   q=points[idx]
   if isnan(q.z);out[idx]=n4(0,0,0,0);return nothing;end
   x=(idx-1)%W;y=(idx-1)÷W
   west=x>1&&!isnan(points[idx-1].z)&&abs(points[idx-1].z-q.z)<200f0
   east=x<W-1&&!isnan(points[idx+1].z)&&abs(points[idx+1].z-q.z)<200f0
   north=y>1&&!isnan(points[idx-W].z)&&abs(points[idx-W].z-q.z)<200f0
   south=y<H-1&&!isnan(points[idx+W].z)&&abs(points[idx+W].z-q.z)<200f0
   if !(west||east)||!(north||south);out[idx]=n4(0,0,0,1);return nothing;end
   h=west&&east ? sub3(points[idx+1],points[idx-1]) : west ? sub3(q,points[idx-1]) : sub3(points[idx+1],q)
   v=north&&south ? sub3(points[idx-W],points[idx+W]) : south ? sub3(q,points[idx+W]) : sub3(points[idx-W],q)
   a=muladd(h[2],v[3],-h[3]*v[2]);b=muladd(h[3],v[1],-h[1]*v[3]);c=muladd(h[1],v[2],-h[2]*v[1]);nn=muladd(a,a,muladd(b,b,c*c));invlen=inv(sqrt(nn));a*=invlen;b*=invlen;c*=invlen
   if muladd(q.x,a,muladd(q.y,b,q.z*c))>0f0;a=-a;b=-b;c=-c;end
   flag=abs(h[3])>0.04f0||abs(v[3])>0.04f0||!west||!east||!north||!south
   out[idx]=n4(a,b,c,flag)
  end
 end
 nothing
end
