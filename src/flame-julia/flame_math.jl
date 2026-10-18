struct VariationParameter
 idx::Int32
 factor::Float32
end
struct FlameParams
 variations::NTuple{100,VariationParameter}
 colors::NTuple{20,Float32}
 pre::NTuple{120,Float32}
 post::NTuple{120,Float32}
 frame::Int32
 sierpinski::Int32
 mapping::NTuple{20,Int32}
end
@assert sizeof(FlameParams)==1928
@assert [fieldoffset(FlameParams,i) for i in 1:7]==[0,800,880,1360,1840,1844,1848]
@generated function params_to_words!(out,p::FlameParams)
 ops=Expr[]
 for i in 1:100
  push!(ops,:(out[$(2i-1)]=reinterpret(UInt32,p.variations[$i].idx)))
  push!(ops,:(out[$(2i)]=reinterpret(UInt32,p.variations[$i].factor)))
 end
 for (field,n,offset) in [(:colors,20,200),(:pre,120,220),(:post,120,340)]
  for i in 1:n;push!(ops,:(out[$(offset+i)]=reinterpret(UInt32,getfield(p,$(QuoteNode(field)))[$i])));end
 end
 push!(ops,:(out[461]=reinterpret(UInt32,p.frame)));push!(ops,:(out[462]=reinterpret(UInt32,p.sierpinski)))
 for i in 1:20;push!(ops,:(out[$(462+i)]=reinterpret(UInt32,p.mapping[$i])));end
 quote
  @inbounds begin
   $(ops...)
  end
  nothing
 end
end
@inline function function_idx(idx,p)
 index=idx & ~Int32(31);lo=Int32(0);hi=Int32(20);f=Int32(0)
 for k in 1:6
  f=(lo+hi)÷Int32(2)
  @inbounds if p.mapping[f+1]<=index;lo=f+Int32(1);else;hi=f;end
 end
 f
end
@inline function affine_point(x,y,a,j)
 k=j*Int32(6)
 @inbounds xx=muladd(a[k+1],x,a[k+2]*y)+a[k+3]
 @inbounds yy=muladd(a[k+4],x,a[k+5]*y)+a[k+6]
 xx,yy
end
@inline function iterate_point(x,y,color,func,idx,randoms,p)
 if p.sierpinski==1 || p.sierpinski==2
  k=p.sierpinski==1 ? idx : func
  x*=.5f0;y*=.5f0
  if k%Int32(3)==1;x+=.5f0;elseif k%Int32(3)==2;y+=.5f0;end
 end
 x,y=affine_point(x,y,p.pre,func)
 radius=sqrt(x*x+y*y);theta=atan(x,y)
 ox=0f0;oy=0f0
 for slot in Int32(1):Int32(5)
  @inbounds vp=p.variations[func*Int32(5)+slot]
  factor=vp.factor
  if abs(factor)<.01f0;continue;end
  variation=vp.idx
  if variation==0
   vx=x;vy=y
  elseif variation==5
   vx=theta/3.1415926535f0;vy=radius-1f0
  elseif variation==7
   vx=radius*sin(theta*radius);vy=-radius*cos(theta*radius)
  elseif variation==9
   vx=radius*(cos(theta)+sin(radius));vy=radius*(sin(theta)-cos(radius))
  elseif variation==13
   @inbounds omega=randoms[idx+1]>.5f0 ? 0f0 : 3.1415926535f0
   phase=theta*.5f0+omega;root=sqrt(radius);vx=root*cos(phase);vy=root*sin(phase)
  else
   # Host contract rejects any active variation outside0,5,7,9,13.
   vx=Float32(NaN);vy=Float32(NaN)
  end
  ox=muladd(factor,vx,ox);oy=muladd(factor,vy,oy)
 end
 @inbounds color=(color+p.colors[func+1])*.5f0
 x,y=affine_point(ox,oy,p.post,func)
 x,y,color
end
@inline function initialize_site!(idx,points,colors,perms,permnum,start,randoms,p)
 func=function_idx(idx,p)
 @inbounds q=Int32(perms[256*permnum+idx+1]);xy=start[q+1]
 x,y,c=iterate_point(xy[1],xy[2],.5f0,func,idx,randoms,p)
 @inbounds points[idx+1]=(Float16(x),Float16(y));colors[idx+1]=Float16(c)
 nothing
end
@inline function iterate_site!(idx,points,colors,perms,permnum,randoms,p)
 func=function_idx(idx,p)
 @inbounds q=Int32(perms[256*permnum+idx+1]);xy=points[q+1];c=Float32(colors[q+1])
 x,y,c=iterate_point(Float32(xy[1]),Float32(xy[2]),c,func,idx,randoms,p)
 @inbounds points[q+1]=(Float16(x),Float16(y));colors[q+1]=Float16(c)
 nothing
end
@inline function generate_site!(idx,vertices,points,colors,perms,permnum,randoms,p)
 func=function_idx(idx,p)
 for k in Int32(0):Int32(63)
  @inbounds q=Int32(perms[((permnum+k)%Int32(128))*Int32(256)+idx+1]);xy=points[q+1];c=Float32(colors[q+1])
  x,y,c=iterate_point(Float32(xy[1]),Float32(xy[2]),c,func,idx,randoms,p)
  @inbounds vertices[idx+k*Int32(256)+1]=(x,y,c)
 end
 nothing
end
