using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","lbm_math.jl"))
@kernel unsafe_indices=true function lbm_kernel!(a,b,v,f,d,w,n,omega,lid)
 I=@index(Global,NTuple)
 lbm_cell!(a,b,v,f,d,w,Int32(I[1]-1),Int32(I[2]-1),Int32(I[3]-1),n,omega,lid)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const LBM_KA=lbm_kernel!(CUDABackend(),(64,1,1))
lbm_study((a,b,v,f,d,w,n,omega,lid)->LBM_KA(a,b,v,f,d,w,n,omega,lid;ndrange=(64,17,17)))
