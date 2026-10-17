include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(data,kernel,n)
 return ()->begin
  fwt_gpu!(data,16);fwt_gpu!(kernel,16)
  @cuda threads=256 blocks=256 modulate_kernel!(data,kernel,Int32(n),Float32(1/n))
  fwt_gpu!(data,16)
 end
end
fwt_study(factory)
