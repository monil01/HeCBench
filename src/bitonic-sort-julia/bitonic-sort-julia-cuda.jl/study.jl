include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(data,n)
    (seq,power)->(@cuda threads=256 blocks=n÷256 bitonic_sort_kernel!(data,seq,power,Int32(n)))
end
exit(study_main(study_prepare))
