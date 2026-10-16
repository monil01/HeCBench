source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch_ref!(::Val{M},a,b,c,m,n) where M
    if M==3
        @cuda threads=128 blocks=(cld(n,256),cld(m,16)) tv_add_kernel!(a,b,c,Int32(m),Int32(n))
    else
        @cuda threads=256 blocks=cld(m*n,256) add_kernel!(a,b,c,Int32(m*n))
    end
end
vadd_study(launch_ref!)
