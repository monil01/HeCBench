using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),"exit(main())"=>"")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(a,b,c,nx,ny,nz,r)
    function launch!(out,inp)
        @cuda threads=(8,8,4) blocks=(cld(nx,8),cld(ny,8),cld(nz,4)) fdtd_kernel!(out,inp,c,Int32(nx),Int32(ny),Int32(nz),Int32(r))
        nothing
    end
    launch!
end
exit(study_main(prepare))
