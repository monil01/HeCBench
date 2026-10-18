include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(f)
 b,h,q,c,pad=f[6:10];rows=Int64(b)*Int64(h)*Int64(q)
 @cuda threads=256 blocks=rows scaled_masked_softmax_kernel!(f[3],f[1],f[2],f[5],rows,c,q,pad,h)
end
study_main("cuda_jl",launch!)
