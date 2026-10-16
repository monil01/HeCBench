using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"\nmain()\n"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(output,input,rows,cols)=()->(@cuda threads=256 blocks=cld(Int(rows),256) softmax_kernel!(output,input,rows,cols))
softmax_study(factory)
