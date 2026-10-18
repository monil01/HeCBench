
using CUDA,JACC,Printf
JACC.@init_backend
@inline function minisweep_add!(a,i,v)
 JACC.@atomic a[i] += v
 nothing
end
include(joinpath(@__DIR__,"..","study_device.jl"))
function init_xy!(i,j,o,d,F)
 ms_init!(Val(1),Int32(i-1),Int32(j-1),Int32(o-1),d,F)
 nothing
end
function init_xz!(i,j,o,d,F)
 ms_init!(Val(2),Int32(i-1),Int32(j-1),Int32(o-1),d,F)
 nothing
end
function init_yz!(i,j,o,d,F)
 ms_init!(Val(3),Int32(i-1),Int32(j-1),Int32(o-1),d,F)
 nothing
end
function sweep!(o,e,d,A,M,vi,vo,xy,xz,yz,angular)
 ms_sweep!(Int32(o-1),Int32(e-1),d,A,M,vi,vo,xy,xz,yz,angular)
 nothing
end
function factory(d,buffers)
 A,M,vi,vo,xy,xz,yz,angular=buffers;nx,ny,nz,ne,nm,na=d
 xygrid=(cld(Int(nx),8),cld(Int(ny),8),2);xzgrid=(cld(Int(nx),8),cld(Int(nz),8),2);yzgrid=(cld(Int(ny),8),cld(Int(nz),8),2);wavegrid=(2,cld(Int(ne),64))
 sxy=JACC.launch_spec(;threads=(8,8,4),blocks=xygrid,shmem_size=0,sync=false)
 sxz=JACC.launch_spec(;threads=(8,8,4),blocks=xzgrid,shmem_size=0,sync=false)
 syz=JACC.launch_spec(;threads=(8,8,4),blocks=yzgrid,shmem_size=0,sync=false)
 sw=JACC.launch_spec(;threads=(4,64),blocks=wavegrid,shmem_size=0,sync=false)
 rxy=(8xygrid[1],8xygrid[2],8);rxz=(8xzgrid[1],8xzgrid[2],8);ryz=(8yzgrid[1],8yzgrid[2],8);rw=(8,64wavegrid[2])
 function launch()
  JACC.parallel_for(sxy,rxy,init_xy!,d,xy);JACC.parallel_for(sxz,rxz,init_xz!,d,xz);JACC.parallel_for(syz,ryz,init_yz!,d,yz)
  JACC.parallel_for(sw,rw,sweep!,d,A,M,vi,vo,xy,xz,yz,angular)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
minisweep_study(factory)
