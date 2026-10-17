include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(src,dst,n)
    ()->(@cuda threads=128 blocks=n÷512 shmem=528*4 split_sort_kernel!(src,dst,Int32(n÷512)))
end
exit(study_main(study_prepare))
