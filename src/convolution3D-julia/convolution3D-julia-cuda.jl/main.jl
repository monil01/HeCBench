include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run!(X,W,Y,dims,::Val{STYLE}) where STYLE
 kernel=STYLE==1 ? conv3d_s1_kernel! : STYLE==2 ? conv3d_s2_kernel! : conv3d_s3_kernel!
 @cuda threads=(16,16,1) blocks=grid(STYLE) kernel(X,W,Y,dims...)
 return
end
study(run!)
