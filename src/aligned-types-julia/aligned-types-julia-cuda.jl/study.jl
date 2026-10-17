include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"main\(\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(dst,src,n)
    ()->(@cuda threads=256 blocks=cld(n,256) copy_kernel!(dst,src,Int32(n)))
end
exit(study_main(study_prepare))
