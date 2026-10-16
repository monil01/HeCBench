using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
source=replace(source,r"\nmain\(\)\s*$"=>"\n")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(a,b,Lx,Ly,delta,norm)
    function launch!(out,inp)
        @cuda threads=256 blocks=cld(Lx*Ly,256) lapl_kernel!(out,inp,delta,norm,Int32(Lx),Int32(Ly))
        return nothing
    end
    launch!
end
exit(study_main(prepare))
