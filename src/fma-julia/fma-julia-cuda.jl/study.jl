using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"exit(main(ARGS))")),joinpath(@__DIR__,"main.jl"))
function study_launch!(a,b,c1,c2,ins,outs,ops,ch,na,nc)
    # Preserve inherited scalar kernel and rowwise alias; native vectorized/shared mapping disclosed.
    @cuda threads=256 blocks=cld(ops*ch,256) implicit_fma_kernel!(a,b,c1,ins,outs,ops,ch)
    @cuda threads=256 blocks=cld(ops*ch,256) implicit_fma_kernel!(a,b,c2,ins,outs,ops,ch)
end
include(joinpath(@__DIR__,"..","study_common.jl"));fma_study(study_launch!)
