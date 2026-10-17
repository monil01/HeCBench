include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(o1,o2,a,b,c,s,n)
    ()->(@cuda threads=128 blocks=cld(n,512) rotary_kernel!(o1,o2,a,b,c,s,Int64(n)))
end
exit(study_main(study_prepare))
