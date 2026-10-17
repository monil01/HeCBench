using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","floyd_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function floyd_kernel!(distance,path,n,k)
    ix,iy=@index(Global,NTuple);x=Int32(ix)-Int32(1);y=Int32(iy)-Int32(1)
    floyd_update!(x,y,distance,path,n,k)
end
function study_prepare(distance,path,n)
    kernel=floyd_kernel!(CUDABackend(),(16,16))
    k->kernel(distance,path,Int32(n),Int32(k);ndrange=(n,n))
end
exit(study_main(study_prepare))
