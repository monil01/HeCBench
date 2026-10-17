include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(out,p,s,lse,pl,sl,tokens,heads,size)
    n=tokens*heads*size;()->(@cuda threads=256 blocks=cld(n,256) merge_kernel!(out,p,s,lse,pl,sl,Int32(n),Int32(size)))
end
exit(study_main(study_prepare))
