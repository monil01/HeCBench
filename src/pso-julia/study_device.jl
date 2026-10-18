using StaticArrays
@inline sourceF(x)=1f0+(x-1f0)/4f0
@inline function source_fitness(x)
 y1=sourceF(x[1]);yn=sourceF(x[30]);s=sin(3.1415f0*y1);z=yn-1f0;res=s*s+z*z
 for j in 1:29
  y=sourceF(x[j]);yp=sourceF(x[j+1]);s=sin(3.1415f0*yp);z=y-1f0;res+=z*z*(1f0+10f0*s*s)
 end
 res
end
@inline function move_particle_element(i,pos,vel,best,globalbest,rp,rg)
 @inbounds begin
  j=(i-Int32(1))%Int32(30)+Int32(1);v=.5f0*vel[i]+1.5f0*rp*(best[i]-pos[i])+1.5f0*rg*(globalbest[j]-pos[i]);vel[i]=v;pos[i]+=v
 end
 return
end
@inline function candidate(i,pos,best)
 a=MVector{30,Float32}(undef);b=MVector{30,Float32}(undef);base=(i-Int32(1))*Int32(30)
 @inbounds for j in 1:30;a[j]=pos[base+j];b[j]=best[base+j];end
 a,source_fitness(a)<source_fitness(b)
end
