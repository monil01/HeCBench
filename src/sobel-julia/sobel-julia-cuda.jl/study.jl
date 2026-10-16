include(joinpath(@__DIR__, "..", "study_common.jl"))
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(source,r"main\(\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(din,dout,w,h)
    ()->(@cuda threads=(16,16) blocks=(32,32) sobel_kernel!(din,dout,Int32(w),Int32(h)))
end
exit(study_main(study_prepare))
