include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_prepare(lens,seeds,keys,out,n)
    ()->(@cuda threads=256 blocks=cld(n,256) jenkins_kernel!(lens,seeds,keys,out,Int64(n)))
end
exit(study_main(study_prepare))
