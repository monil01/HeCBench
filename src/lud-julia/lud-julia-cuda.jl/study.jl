include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(m,n)
 for off in 0:16:n-32
  tail=(n-off)÷16-1
  @cuda threads=16 blocks=1 diagonal_kernel!(m,n,off)
  @cuda threads=32 blocks=tail perimeter_kernel!(m,n,off)
  @cuda threads=(16,16) blocks=(tail,tail) internal_kernel!(m,n,off)
 end
 @cuda threads=16 blocks=1 diagonal_kernel!(m,n,n-16)
end
study_main(study_launch!)
