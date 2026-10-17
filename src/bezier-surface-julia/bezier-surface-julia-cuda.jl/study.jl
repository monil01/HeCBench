using CUDA
text=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(text,r"\nexit\(main\(ARGS\)\)\s*$"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
bezier_study((input,output,n)->()->(@cuda threads=256 blocks=2 bezier_gpu_kernel(input,output,Int32(3),Int32(3),n,n)))
