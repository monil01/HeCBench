using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
# A distinct CPU scalar implementation retains the native piecewise alpha (including its a1 middle-interval coefficient).
function aidw_cpu(dx,dy,dz,ix,iy,avg,area)
    out=zeros(Float32,length(ix))
    for t in eachindex(ix)
        r=avg[t]/(0.5f0/sqrt(Float32(length(dx))/area))
        u=r>=2f0 ? 1f0 : r>=0f0 ? 0.5f0-0.5f0*cos(3.1415926f0/2f0*r) : 0f0
        alpha = u<=0.1f0 ? 1.5f0 : u<=0.3f0 ? 1.5f0*(1f0-5f0*(u-0.1f0))+2f0*5f0*(u-0.1f0) : u<=0.5f0 ? 2.5f0*5f0*(u-0.3f0)+1.5f0*(1f0-5f0*(u-0.3f0)) : u<=0.7f0 ? 2.5f0*(1f0-5f0*(u-0.5f0))+3f0*5f0*(u-0.5f0) : u<=0.9f0 ? 3.5f0*5f0*(u-0.7f0)+3f0*(1f0-5f0*(u-0.7f0)) : 3.5f0
        s=0f0;z=0f0
        for j in eachindex(dx)
            xx=ix[t]-dx[j];yy=iy[t]-dy[j];w=1f0/((xx*xx+yy*yy)^(alpha*0.5f0))
            s+=w;z+=dz[j]*w
        end
        out[t]=z/s
    end
    out
end
function aidw_study(factory)
    n=1024;area=4000000f0;repeats=10
    dx=Float32[16*(i%32)+1 for i in 0:n-1];dy=Float32[16*(i÷32)+2 for i in 0:n-1];dz=Float32[11f0+Float32(i%47)/8f0 for i in 0:n-1]
    ix=Float32[16*(i%32)+7.25f0 for i in 0:n-1];iy=Float32[16*(i÷32)+9.5f0 for i in 0:n-1];avg=Float32[2*(i%33) for i in 0:n-1]
    expected=aidw_cpu(dx,dy,dz,ix,iy,avg,area)
    @assert all(x->minimum(dz)<=x<=maximum(dz),expected)
    iz=CUDA.zeros(Float32,n)
    launch! = factory(CuArray(dx),CuArray(dy),CuArray(dz),Int32(n),CuArray(ix),CuArray(iy),iz,Int32(n),area,CuArray(avg))
    OwnedStudy.run_trials!(launch!,()->fill!(iz,0f0),()->Array(iz),expected,repeats;atol=1e-3,rtol=0.)
end
