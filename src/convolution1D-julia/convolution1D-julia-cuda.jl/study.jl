include("main.jl")
include("study_kernels.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(a,o,m,mw,mode::Val{MODE},::Val{BT}) where {MODE,BT}
 if MODE==1
  @cuda threads=BT blocks=(2048÷BT) conv1d_kernel!(a,o,m,Int32(2048),mw)
 else
  @cuda threads=BT blocks=(2048÷BT) shmem=((BT+(MODE==2 ? 9 : 0))*sizeof(eltype(a))) conv_tiled_kernel!(a,o,m,Int32(2048),mw,mode)
 end
end
study_main(study_launch!)
