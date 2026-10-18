using CUDA
include("../reaction_device.jl")
function derivative!(axis,f,out,M)
 shared=CUDA.CuDynamicSharedArray(Float32,2*Int(M))
 i,slot,coordinate,step=pencil_position(axis,threadIdx().x,threadIdx().y,blockIdx().x,blockIdx().y,M)
 @inbounds shared[slot]=f[i]
 CUDA.sync_threads()
 @inbounds out[i]=pencil_difference(shared,slot,coordinate,step,M)
 nothing
end
function point_kernel!(op,out,d,N,coef,dt,c1,c2)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 point_site!(op,out,d,i,N,coef,dt,c1,c2)
 nothing
end
function diff_launch(axis::Val{A},f,out,c) where A
 M=Int(c.M);threads=A==1 ? (M,2) : (2,M)
 @cuda threads=threads blocks=(M÷2,M) shmem=2*M*sizeof(Float32) derivative!(axis,f,out,c.M)
 nothing
end
function point_launch(op,out,c,coef)
 @cuda threads=Int(c.M) blocks=Int(c.N÷c.M) point_kernel!(op,out,c.d,c.N,coef,c.dt,c.c1,c.c2)
 nothing
end
include("../study_common.jl")
reaction_study(diff_launch,point_launch)
