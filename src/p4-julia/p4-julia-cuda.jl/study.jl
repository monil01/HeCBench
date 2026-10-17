include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(d,threshold)
 @cuda threads=ST_NA blocks=ST_FX*ST_FY postprocess_kernel!(d...,ST_RANGES...,ST_FX,ST_FY,ST_NA,ST_NC,ST_NB,threshold,ST_OFFSET)
end
study_main("cuda_jl",launch!)
