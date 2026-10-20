using CUDA,Printf,SHA
include(joinpath(@__DIR__,"..","columnar_math.jl"))
function setup_kernel!(state)
    idx=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);s=UInt32(idx)
    for i in Int32(0):idx-Int32(1);s=cs_step(s);end
    @inbounds state[idx+1]=s
    return
end
function decode_kernel!(scores,encrypted,seeds,output,diag)
    shared=CuStaticSharedArray(Float32,676)
    if threadIdx().x==1
        @inbounds for j in 1:676;shared[j]=scores[j];end
    end
    sync_threads()
    idx=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
    cs_decode_site!(idx,shared,encrypted,seeds,output,diag)
    return
end
function launch_both!(scores,cipher,seeds,out,diag)
    @cuda threads=32 blocks=32 setup_kernel!(seeds)
    @cuda threads=32 blocks=32 decode_kernel!(scores,cipher,seeds,out,diag)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
