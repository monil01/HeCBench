struct PICParticle
 x::Float64
 v::Float64
 alive::Bool
end
@assert sizeof(PICParticle)==24 && fieldoffset(PICParticle,3)==16
@inline function logical_position(x)
 x/1e-4
end
@inline function field_gather(x,ef)
 lc=logical_position(x);i=unsafe_trunc(Int32,lc);d=lc-Float64(i)
 @inbounds value=muladd(ef[i+1],1.0-d,ef[i+2]*d)
 value
end
@inline function body_scatter!(i,particles,den,n)
 if i<n
  @inbounds p=particles[i+1]
  if p.alive
   lc=logical_position(p.x);j=unsafe_trunc(Int32,lc);d=Float32(lc-Float64(j))
   density_add!(den,j+Int32(1),1f0-d);density_add!(den,j+Int32(2),d)
  end
 end
 nothing
end
@inline function body_push!(i,particles,ef,qm,n)
 if i<n
  @inbounds p=particles[i+1]
  if p.alive
   e=field_gather(p.x,ef);v=muladd(1e-11*qm,e,p.v);x=muladd(1e-11,v,p.x);alive=!(x<0.0||x>=100*1e-4)
   @inbounds particles[i+1]=PICParticle(x,v,alive)
  end
 end
 nothing
end
@inline function body_rewind!(i,particles,ef,qm,n)
 if i<n
  @inbounds p=particles[i+1]
  if p.alive
   e=field_gather(p.x,ef);v=muladd(-.5e-11*qm,e,p.v)
   @inbounds particles[i+1]=PICParticle(p.x,v,p.alive)
  end
 end
 nothing
end
