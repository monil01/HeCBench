include_string(@__MODULE__,replace(read(joinpath(@__DIR__,"main.jl"),String),"\nmain()\n"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(out,buffer,input,kernel,w,h)=()->run_gpu!(out,buffer,input,kernel,w,h)
conv_study(factory)
