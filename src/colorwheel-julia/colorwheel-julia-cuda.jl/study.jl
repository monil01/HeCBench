include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(pix,size,half,range,truerange)
    ()->(@cuda threads=(16,16) blocks=(32,32) color_kernel!(pix,Int32(size),Int32(half),range,truerange))
end
exit(study_main(study_prepare))
