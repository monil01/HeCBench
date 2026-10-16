using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(input,output,n,vec_size)=()->(@cuda threads=256 blocks=cld(n,256*Int(vec_size)) thread_copy_kernel!(input,output,n,vec_size))
copy_study(factory)
