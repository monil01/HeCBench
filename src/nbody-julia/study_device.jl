# Intel nbody source Particle ABI: float3 position, velocity, acceleration, mass.
struct StudyParticle
    pos::NTuple{3,Float32}
    vel::NTuple{3,Float32}
    acc::NTuple{3,Float32}
    mass::Float32
end
Base.zero(::Type{StudyParticle}) = StudyParticle((0f0,0f0,0f0),(0f0,0f0,0f0),(0f0,0f0,0f0),0f0)
const PARTICLE_GUARD=StudyParticle((-8192.25f0,-8192.25f0,-8192.25f0),(-8192.25f0,-8192.25f0,-8192.25f0),(-8192.25f0,-8192.25f0,-8192.25f0),-8192.25f0)
@assert sizeof(StudyParticle)==40
@inline function accelerate_body!(p,n::Int32,soft::Float32,g::Float32,i::Int32)
    if i<n
        pi=@inbounds p[i+Int32(1)]
        a0,a1,a2=pi.acc
        for j in Int32(1):n
            pj=@inbounds p[j]
            dx=pj.pos[1]-pi.pos[1]
            dy=pj.pos[2]-pi.pos[2]
            dz=pj.pos[3]-pi.pos[3]
            r2=dx*dx+dy*dy+dz*dz+soft
            # Only this expression uses fast math, matching native rsqrtf's approximate SFU.
            invr=@fastmath inv(sqrt(r2))
            strength=g*pj.mass*invr*invr*invr
            a0+=dx*strength;a1+=dy*strength;a2+=dz*strength
        end
        @inbounds p[i+Int32(1)]=StudyParticle(pi.pos,pi.vel,(a0,a1,a2),pi.mass)
    end
    return
end
@inline function update_body!(p,energy,n::Int32,dt::Float32,i::Int32)
    if i<n
        pi=@inbounds p[i+Int32(1)]
        v0=pi.vel[1]+pi.acc[1]*dt
        v1=pi.vel[2]+pi.acc[2]*dt
        v2=pi.vel[3]+pi.acc[3]*dt
        x0=pi.pos[1]+v0*dt
        x1=pi.pos[2]+v1*dt
        x2=pi.pos[3]+v2*dt
        @inbounds energy[i+Int32(1)]=pi.mass*(v0*v0+v1*v1+v2*v2)
        @inbounds p[i+Int32(1)]=StudyParticle((x0,x1,x2),(v0,v1,v2),(0f0,0f0,0f0),pi.mass)
    end
    return
end
@inline function accumulate_body!(energy,n::Int32)
    # Native one-thread serial Float32 sum; e[1] becomes global twice kinetic energy.
    @inbounds for i in Int32(2):n
        energy[1]+=energy[i]
    end
    return
end
