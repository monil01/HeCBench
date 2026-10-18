using CUDA,KernelAbstractions
include("../reaction_device.jl")
@kernel unsafe_indices=true function derivative!(axis,f,out,M,::Val{CAP}) where CAP
 shared=@localmem Float32 (CAP,)
 localindex=@index(Local,NTuple)
 groupindex=@index(Group,NTuple)
 i,slot,coordinate,step=pencil_position(axis,localindex[1],localindex[2],groupindex[1],groupindex[2],M)
 @inbounds shared[slot]=f[i]
 @synchronize
 @inbounds out[i]=pencil_difference(shared,slot,coordinate,step,M)
end
@kernel unsafe_indices=true function point_kernel!(op,out,d,N,coef,dt,c1,c2)
 point_site!(op,out,d,@index(Global,Linear),N,coef,dt,c1,c2)
end
function diff_launch(axis::Val{A},f,out,c) where A
 M=Int(c.M);threads=A==1 ? (M,2) : (2,M);grid=(M÷2,M)
 derivative!(CUDABackend(),threads)(axis,f,out,c.M,Val(2*M);ndrange=(threads[1]*grid[1],threads[2]*grid[2]))
 nothing
end
function point_launch(op,out,c,coef)
 point_kernel!(CUDABackend(),Int(c.M))(op,out,c.d,c.N,coef,c.dt,c.c1,c.c2;ndrange=Int(c.N))
 nothing
end
include("../study_common.jl")
reaction_study(diff_launch,point_launch)
