using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),r"\nmain\(\)\s*$"=>"\n")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(a,b,err)
    function launch!(out,inp,e)
        @cuda threads=(16,16) blocks=(128,128) jacobi_kernel!(out,inp,e,Int32(STUDY_N))
        nothing
    end
    launch!
end
exit(study_main(prepare))
