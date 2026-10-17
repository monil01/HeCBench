include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(input,hidden,b1,b2,hx,hy,storage,h,n)
    ()->(@cuda threads=256 blocks=cld(n,256) gru_cell_forward!(input,hidden,b1,b2,hx,hy,storage,Int32(h),Int32(n)))
end
exit(study_main(study_prepare))
