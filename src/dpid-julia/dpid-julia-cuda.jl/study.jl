using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
ending=findlast("\nmain()",source)
ending===nothing&&error("Missing originalmain guard")
include_string(Main,source[1:first(ending)-1],joinpath(@__DIR__,"main.jl"))
include("../study_common.jl")
function factory(input,guidance,output,p)
 blocks=cld(Int(p.o_width)*Int(p.o_height),128)
 ()->begin
  CUDA.@cuda threads=128 blocks=blocks guidance_kernel!(input,guidance,p)
  CUDA.@cuda threads=128 blocks=blocks downsample_kernel!(input,guidance,output,p)
 end
end
dpid_study(factory)
