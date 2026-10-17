include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(run_overlap\(\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(din,dout,n,inner)
    s->(@cuda threads=256 blocks=cld(n,256) inc_kernel!(dout[s],din[s],Int32(n),inner))
end
exit(study_main(study_prepare))
