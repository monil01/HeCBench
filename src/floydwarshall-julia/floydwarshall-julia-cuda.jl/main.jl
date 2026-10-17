using CUDA
include(joinpath(@__DIR__,"..","floyd_math.jl"))
function floyd_kernel!(distance,path,n,k)
    x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y)-Int32(1)
    floyd_update!(x,y,distance,path,n,k)
end
function study_prepare(distance,path,n)
    k->(@cuda threads=(16,16) blocks=(n÷16,n÷16) floyd_kernel!(distance,path,Int32(n),Int32(k)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
