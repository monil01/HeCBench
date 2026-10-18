using CUDA,StaticArrays,KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function column_kernel(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18)
 l=@index(Local,NTuple);g=@index(Group,NTuple)
 wsm_body!(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,UInt32(g[1]-1),UInt32(g[2]-1),UInt32(8),UInt32(8),UInt32(l[1]-1),UInt32(l[2]-1))
end
function launch(c)
 column_kernel(BACKEND,(8,8))(c.d...,c.dt,c.bounds...;ndrange=(cld(Int(c.nx),8)*8,cld(Int(c.ny),8)*8))
end
function alloc(v)
 a=KernelAbstractions.allocate(BACKEND,Float32,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(alloc,launch)
