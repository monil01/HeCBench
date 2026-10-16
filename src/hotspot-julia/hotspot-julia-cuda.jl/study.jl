using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),r"
main\(\)\s*$"=>"
")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(a,b,p,dc,rx,ry,rz)
    function launch!(out,inp)
        @cuda threads=(16,16) blocks=(32,32) hotspot_kernel!(out,inp,p,dc,rx,ry,rz)
        nothing
    end
    launch!
end
exit(study_main(prepare))
