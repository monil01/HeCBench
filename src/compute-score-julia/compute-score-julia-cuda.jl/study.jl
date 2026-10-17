include_string(@__MODULE__,replace(read(joinpath(@__DIR__,"main.jl"),String),"exit(main())"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(a,z,w1,w2,bh,hi,lo,info,out,lines)
return ()->begin
@cuda threads=8 blocks=lines compute_kernel!(a,z,w1,w2,bh,hi,lo)
@cuda threads=64 blocks=1 reduction_kernel!(info,hi,lo,out)
end
end
score_study(factory)
