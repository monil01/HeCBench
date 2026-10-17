include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(ARGS\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(y,a,b,d,o1,o2,n,m)
    function launch!(mode)
        if mode==1
            @cuda threads=256 blocks=cld(n,256) mrc_gradient_kernel!(Int32(n),y,a,b,d,m,o1,o2)
        elseif mode==2
            @cuda threads=256 blocks=cld(n,256) mrc_gradient2_kernel!(Int32(n),y,a,b,d,m,o1,o2)
        else
            @cuda threads=256 blocks=cld(n÷4,256) mrc_gradient3_kernel!(Int32(n),y,a,b,d,m,o1,o2)
        end
    end
end
exit(study_main(study_prepare))
