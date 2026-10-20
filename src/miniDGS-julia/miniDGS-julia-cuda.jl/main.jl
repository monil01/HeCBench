using CUDA
function volume_kernel(Q,D,G,rhs)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
s=CUDA.@cuStaticSharedMem(Float32,588)
@inbounds begin
for c in 0:5;s[c*96+tid+1]=Q[gid*576+c*96+tid+1];end
if tid<12;s[576+tid+1]=G[gid*12+tid+1];end
CUDA.sync_threads()
d00=0f0
d01=0f0
d02=0f0
d10=0f0
d11=0f0
d12=0f0
d20=0f0
d21=0f0
d22=0f0
d30=0f0
d31=0f0
d32=0f0
d40=0f0
d41=0f0
d42=0f0
d50=0f0
d51=0f0
d52=0f0
for m in 0:83
 dr=D[4*(tid+m*96)+1];ds=D[4*(tid+m*96)+2];dt=D[4*(tid+m*96)+3]
q=s[0*96+m+1]
d00=muladd(dr,q,d00)
d01=muladd(ds,q,d01)
d02=muladd(dt,q,d02)
q=s[1*96+m+1]
d10=muladd(dr,q,d10)
d11=muladd(ds,q,d11)
d12=muladd(dt,q,d12)
q=s[2*96+m+1]
d20=muladd(dr,q,d20)
d21=muladd(ds,q,d21)
d22=muladd(dt,q,d22)
q=s[3*96+m+1]
d30=muladd(dr,q,d30)
d31=muladd(ds,q,d31)
d32=muladd(dt,q,d32)
q=s[4*96+m+1]
d40=muladd(dr,q,d40)
d41=muladd(ds,q,d41)
d42=muladd(dt,q,d42)
q=s[5*96+m+1]
d50=muladd(dr,q,d50)
d51=muladd(ds,q,d51)
d52=muladd(dt,q,d52)
end
rx=s[576+0+1]
ry=s[576+1+1]
rz=s[576+2+1]
sx=s[576+4+1]
sy=s[576+5+1]
sz=s[576+6+1]
tx=s[576+8+1]
ty=s[576+9+1]
tz=s[576+10+1]
rhs[gid*576+0*96+tid+1]=Float32(-1*(ry*d50+sy*d51+ty*d52-rz*d40-sz*d41-tz*d42))
rhs[gid*576+1*96+tid+1]=Float32(-1*(rz*d30+sz*d31+tz*d32-rx*d50-sx*d51-tx*d52))
rhs[gid*576+2*96+tid+1]=Float32(-1*(rx*d40+sx*d41+tx*d42-ry*d30-sy*d31-ty*d32))
rhs[gid*576+3*96+tid+1]=Float32(1*(ry*d20+sy*d21+ty*d22-rz*d10-sz*d11-tz*d12))
rhs[gid*576+4*96+tid+1]=Float32(1*(rz*d00+sz*d01+tz*d02-rx*d20-sx*d21-tx*d22))
rhs[gid*576+5*96+tid+1]=Float32(1*(rx*d10+sx*d11+tx*d12-ry*d00-sy*d01-ty*d02))

end
return nothing
end
function surface_kernel(Q,part,L,S,rhs)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
s=CUDA.@cuStaticSharedMem(Float32,672)
@inbounds begin
if tid<112
 off=gid*784+tid;m=Int(S[off+1]);p=Int(S[off+112+1]);Fsc=S[off+224+1];Bsc=S[off+336+1];nx=S[off+448+1];ny=S[off+560+1];nz=S[off+672+1]
 if p<0
  remote=6*(-1-p)
  d0=Float64(Fsc*(part[remote+0+1]-Q[m+0*96+1]))
  d1=Float64(Fsc*(part[remote+1+1]-Q[m+1*96+1]))
  d2=Float64(Fsc*(part[remote+2+1]-Q[m+2*96+1]))
  d3=Float64(Fsc*(part[remote+3+1]-Q[m+3*96+1]))
  d4=Float64(Fsc*(part[remote+4+1]-Q[m+4*96+1]))
  d5=Float64(Fsc*(part[remote+5+1]-Q[m+5*96+1]))
 else
  d0=Float64(Fsc*(Q[p+0*96+1]-Q[m+0*96+1]))
  d1=Float64(Fsc*(Q[p+1*96+1]-Q[m+1*96+1]))
  d2=Float64(Fsc*(Q[p+2*96+1]-Q[m+2*96+1]))
  d3=Float64(Fsc*(Bsc*Q[p+3*96+1]-Q[m+3*96+1]))
  d4=Float64(Fsc*(Bsc*Q[p+4*96+1]-Q[m+4*96+1]))
  d5=Float64(Fsc*(Bsc*Q[p+5*96+1]-Q[m+5*96+1]))
 end
 ndH=Float64(nx)*d0+Float64(ny)*d1+Float64(nz)*d2
 ndE=Float64(nx)*d3+Float64(ny)*d4+Float64(nz)*d5
 s[0*112+tid+1]=Float32(-ny*d5+nz*d4+d0-ndH*nx)
 s[1*112+tid+1]=Float32(-nz*d3+nx*d5+d1-ndH*ny)
 s[2*112+tid+1]=Float32(-nx*d4+ny*d3+d2-ndH*nz)
 s[3*112+tid+1]=Float32(ny*d2-nz*d1+d3-ndE*nx)
 s[4*112+tid+1]=Float32(nz*d0-nx*d2+d4-ndE*ny)
 s[5*112+tid+1]=Float32(nx*d1-ny*d0+d5-ndE*nz)
end
CUDA.sync_threads()
if tid<84
r0=0f0
r1=0f0
r2=0f0
r3=0f0
r4=0f0
r5=0f0
for m in 0:111
 l=L[4*(tid+84*div(m,4))+m%4+1]
r0=muladd(l,s[0*112+m+1],r0)
r1=muladd(l,s[1*112+m+1],r1)
r2=muladd(l,s[2*112+m+1],r2)
r3=muladd(l,s[3*112+m+1],r3)
r4=muladd(l,s[4*112+m+1],r4)
r5=muladd(l,s[5*112+m+1],r5)
end
rhs[gid*576+0*96+tid+1]+=r0
rhs[gid*576+1*96+tid+1]+=r1
rhs[gid*576+2*96+tid+1]+=r2
rhs[gid*576+3*96+tid+1]+=r3
rhs[gid*576+4*96+tid+1]+=r4
rhs[gid*576+5*96+tid+1]+=r5
end

end
return nothing
end
function rk_kernel(Q,res,rhs,fa,fb,fdt)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
@inbounds begin
if i<294336
 value=muladd(fa,res[i+1],fdt*rhs[i+1]);res[i+1]=value;Q[i+1]=muladd(fb,value,Q[i+1])
end

end
return nothing
end
function gather_kernel(Q,map,out)
 tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid
@inbounds begin
if i<12768;out[i+1]=Q[Int(map[i+1])+1];end

end
return nothing
end
alloc_backend(::Type{T},n) where T=CUDA.CuArray{T}(undef,n)
run_gather!(Q,mapping,out)=CUDA.@cuda threads=256 blocks=50 gather_kernel(Q,mapping,out)
run_volume!(Q,D,G,rhs)=CUDA.@cuda threads=84 blocks=511 volume_kernel(Q,D,G,rhs)
run_surface!(Q,part,L,S,rhs)=CUDA.@cuda threads=112 blocks=511 surface_kernel(Q,part,L,S,rhs)
run_rk!(Q,res,rhs,fa,fb,fdt)=CUDA.@cuda threads=256 blocks=1150 rk_kernel(Q,res,rhs,fa,fb,fdt)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
