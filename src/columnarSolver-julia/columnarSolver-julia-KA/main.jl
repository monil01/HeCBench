using CUDA,KernelAbstractions,Printf,SHA
include(joinpath(@__DIR__,"..","columnar_math.jl"))
@kernel function setup_kernel!(state)
    idx=Int32(@index(Global,Linear)-1);s=UInt32(idx)
    for i in Int32(0):idx-Int32(1);s=cs_step(s);end
    @inbounds state[idx+1]=s
end
@kernel function decode_kernel!(scores,encrypted,seeds,output,diag)
    lid=@index(Local,Linear);idx=Int32(@index(Global,Linear)-1)
    shared=@localmem Float32 (676,)
    if lid==1
        @inbounds for j in 1:676;shared[j]=scores[j];end
    end
    @synchronize
    cs_decode_site!(idx,shared,encrypted,seeds,output,diag)
end
function launch_both!(scores,cipher,seeds,out,diag)
    setup_kernel!(get_backend(seeds),32)(seeds;ndrange=1024)
    decode_kernel!(get_backend(scores),32)(scores,cipher,seeds,out,diag;ndrange=1024)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
