using CUDA
using JACC
include(joinpath(@__DIR__,"..","floyd_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function floyd_kernel!(ix,iy,distance,path,n,k)
    x=Int32(ix)-Int32(1);y=Int32(iy)-Int32(1)
    floyd_update!(x,y,distance,path,n,k)
end
function study_prepare(distance,path,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(n÷16,n÷16),shmem_size=0,sync=false)
    k->JACC.parallel_for(floyd_kernel!,spec,(n,n),distance,path,Int32(n),Int32(k))
end
exit(study_main(study_prepare))
