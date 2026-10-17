using CUDA
text=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(text,r"\nexit\(main\(ARGS\)\)\s*$"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
mha_study((q,k,v,beam,steps,qcol,vcol,heads,scale,threshold,out)->()->(@cuda threads=256 blocks=64 shmem=1060 mha_kernel!(q,k,v,beam,steps,qcol,vcol,heads,scale,threshold,out)))
