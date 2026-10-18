using CUDA,StaticArrays
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function column_kernel(xx,yy,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18)
 wsm_body!(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,UInt32(div(xx-1,8)),UInt32(div(yy-1,8)),UInt32(8),UInt32(8),UInt32(mod(xx-1,8)),UInt32(mod(yy-1,8)))
 return
end
function launch(c)
 blocks=(cld(Int(c.nx),8),cld(Int(c.ny),8));spec=JACC.launch_spec(threads=(8,8),blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,(blocks[1]*8,blocks[2]*8),column_kernel,c.d...,c.dt,c.bounds...)
end
function alloc(v)
 a=JACC.zeros(Float32,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(alloc,launch)
