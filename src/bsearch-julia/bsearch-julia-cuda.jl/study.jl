include("main.jl")
include("study_kernels.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(a,z,r,n,mode::Val{MODE}) where {MODE}
 if MODE==1
  @cuda threads=256 blocks=2 bs_kernel!(a,z,r,Int64(512),n)
 elseif MODE==4
  @cuda threads=256 blocks=2 bs_shared_kernel!(a,z,r,Int64(512),n)
 else
  @cuda threads=256 blocks=2 bs_bit_kernel!(a,z,r,Int64(512),n,mode)
 end
end
study_main(study_launch!)
