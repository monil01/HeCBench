include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(rp,ci,av,b,c,nr,bc)
    method->(@cuda threads=(32,4) blocks=(cld(nr,4),cld(bc,32<<(method-1))) spmm_kernel!(Int32(nr),Int32(bc),rp,ci,av,b,c,Int32(method)))
end
exit(study_main(study_prepare))
