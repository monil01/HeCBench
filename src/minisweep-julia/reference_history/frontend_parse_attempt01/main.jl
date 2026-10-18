
using CUDA,Printf
@inline function minisweep_add!(a,i,v)
 CUDA.atomic_add!(pointer(a,i),v);nothing
end
include(joinpath(@__DIR__,"..","study_device.jl"))
function init_xy!(d,F)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);j=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);o=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 ms_init!(Val(1),i,j,o,d,F);return
end
function init_xz!(d,F)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);j=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);o=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 ms_init!(Val(2),i,j,o,d,F);return
end
function init_yz!(d,F)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);j=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);o=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 ms_init!(Val(3),i,j,o,d,F);return
end
function sweep!(d,A,M,vi,vo,xy,xz,yz,local)
 o=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);e=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
 ms_sweep!(o,e,d,A,M,vi,vo,xy,xz,yz,local);return
end
function factory(d,buffers)
 A,M,vi,vo,xy,xz,yz,local=buffers;nx,ny,nz,ne,nm,na=d
 xygrid=(cld(Int(nx),8),cld(Int(ny),8),2);xzgrid=(cld(Int(nx),8),cld(Int(nz),8),2);yzgrid=(cld(Int(ny),8),cld(Int(nz),8),2);wavegrid=(2,cld(Int(ne),64))
 function launch()
  @cuda threads=(8,8,4) blocks=xygrid init_xy!(d,xy)
  @cuda threads=(8,8,4) blocks=xzgrid init_xz!(d,xz)
  @cuda threads=(8,8,4) blocks=yzgrid init_yz!(d,yz)
  @cuda threads=(4,64) blocks=wavegrid sweep!(d,A,M,vi,vo,xy,xz,yz,local)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
minisweep_study(factory)
