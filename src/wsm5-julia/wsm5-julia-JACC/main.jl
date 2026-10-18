using CUDA,StaticArrays
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function column_kernel(xx,yy,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,bounds)
 wsm_body!(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,bounds[1],bounds[2],bounds[3],bounds[4],bounds[5],bounds[6],bounds[7],bounds[8],bounds[9],bounds[10],bounds[11],bounds[12],bounds[13],bounds[14],bounds[15],bounds[16],bounds[17],bounds[18],UInt32(div(xx-1,8)),UInt32(div(yy-1,8)),UInt32(8),UInt32(8),UInt32(mod(xx-1,8)),UInt32(mod(yy-1,8)))
 return
end
function launch(c)
 blocks=(cld(Int(c.nx),8),cld(Int(c.ny),8));spec=JACC.launch_spec(threads=(8,8),blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,(blocks[1]*8,blocks[2]*8),column_kernel,c.d...,c.dt,Tuple(c.bounds))
end
function alloc(v)
 a=JACC.zeros(Float32,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(alloc,launch)
