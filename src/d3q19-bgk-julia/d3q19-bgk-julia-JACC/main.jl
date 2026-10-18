using CUDA,JACC
include(joinpath(@__DIR__,"..","lbm_math.jl"))
function lbm_kernel!(i,j,k,a,b,v,f,d,w,n,omega,lid)
 lbm_cell!(a,b,v,f,d,w,Int32(i-1),Int32(j-1),Int32(k-1),n,omega,lid)
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const LBM_SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(64,1,1),blocks=(1,17,17),shmem_size=0,sync=false)
lbm_study((a,b,v,f,d,w,n,omega,lid)->JACC.parallel_for(lbm_kernel!,LBM_SPEC,(64,17,17),a,b,v,f,d,w,n,omega,lid))
