
using CUDA,KernelAbstractions,Atomix,Printf
@inline function minisweep_add!(a,i,v)
 Atomix.@atomic a[i] += v
 nothing
end
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function init_xy!(d,F)
 I=@index(Global,NTuple)
 ms_init!(Val(1),Int32(I[1]-1),Int32(I[2]-1),Int32(I[3]-1),d,F)
end
@kernel function init_xz!(d,F)
 I=@index(Global,NTuple)
 ms_init!(Val(2),Int32(I[1]-1),Int32(I[2]-1),Int32(I[3]-1),d,F)
end
@kernel function init_yz!(d,F)
 I=@index(Global,NTuple)
 ms_init!(Val(3),Int32(I[1]-1),Int32(I[2]-1),Int32(I[3]-1),d,F)
end
@kernel function sweep!(d,A,M,vi,vo,xy,xz,yz,angular)
 I=@index(Global,NTuple)
 ms_sweep!(Int32(I[1]-1),Int32(I[2]-1),d,A,M,vi,vo,xy,xz,yz,angular)
end
function factory(d,buffers)
 A,M,vi,vo,xy,xz,yz,angular=buffers;nx,ny,nz,ne,nm,na=d
 kxy=init_xy!(CUDABackend(),(8,8,4));kxz=init_xz!(CUDABackend(),(8,8,4));kyz=init_yz!(CUDABackend(),(8,8,4));ks=sweep!(CUDABackend(),(4,64))
 rxy=(cld(Int(nx),8)*8,cld(Int(ny),8)*8,8);rxz=(cld(Int(nx),8)*8,cld(Int(nz),8)*8,8);ryz=(cld(Int(ny),8)*8,cld(Int(nz),8)*8,8);rw=(8,cld(Int(ne),64)*64)
 function launch()
  kxy(d,xy;ndrange=rxy);kxz(d,xz;ndrange=rxz);kyz(d,yz;ndrange=ryz)
  ks(d,A,M,vi,vo,xy,xz,yz,angular;ndrange=rw)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
minisweep_study(factory)
