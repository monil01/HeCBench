using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function reference")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(input,labels,output)=()->(@cuda threads=256 blocks=1024 accuracy_kernel!(Int32(4096),Int32(128),Int32(5),input,labels,output))
accuracy_study(factory)
