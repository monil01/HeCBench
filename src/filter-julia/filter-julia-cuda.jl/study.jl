include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(dst,count,src,n)
    function launch!(mode)
        if mode==0
            @cuda threads=256 blocks=cld(n,256) filter_shared!(dst,count,src,Int32(n))
        else
            @cuda threads=256 blocks=cld(n,256) filter_global!(dst,count,src,Int32(n))
        end
    end
end
exit(study_main(study_prepare))
