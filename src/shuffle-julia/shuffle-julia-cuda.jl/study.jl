include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(kind,s,pattern,out,input)
 if kind==1
  @cuda threads=256 blocks=1 bcast_shfl_xor_kernel!(out,Int32(s))
 elseif kind==2
  @cuda threads=256 blocks=1 bcast_shfl_kernel!(pattern,out,Int32(s))
 else
  @cuda threads=s blocks=STUDY_N÷s transpose_shfl_kernel!(out,input,Int32(s))
 end
end
study_main(launch!)
