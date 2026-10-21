using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),r"\nmain\(\)\s*$"=>"\n")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(din,dout,W,H,a,vi,vs)
    function launch!(radius)
        @cuda threads=(16,16) blocks=(cld(W,16),cld(H,16)) bilateral_filter_kernel!(din,dout,Int32(W),Int32(H),a,vi,vs,radius)
        nothing
    end
    launch!
end
exit(study_main(prepare))
