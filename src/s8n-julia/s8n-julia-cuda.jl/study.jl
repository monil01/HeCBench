include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_prepare(xyz,out,b,n,ranks)
    k=ranks==1 ? cube_select_kernel! : ranks==2 ? cube_select_two_kernel! : cube_select_four_kernel!
    ()->(@cuda threads=512 blocks=b k(Int32(n),Int32(512),xyz,out))
end
exit(study_main(study_prepare))
