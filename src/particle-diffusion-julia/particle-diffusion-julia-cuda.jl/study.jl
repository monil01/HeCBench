include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(x,y,rx,ry,m,n,steps,g,r)
    ()->(@cuda threads=256 blocks=cld(n,256) simulation_kernel!(x,y,rx,ry,m,Int32(n),Int32(steps),Int32(g),r))
end
exit(study_main(study_prepare))
