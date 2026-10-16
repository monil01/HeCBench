using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(out,total,input,target,weights)=()->(@cuda threads=256 blocks=1 shmem=2048 nll_loss_kernel!(out,total,input,target,weights,true,Int64(4096),Int64(64),Int64(32),Val(256)))
nlll_study(factory)
