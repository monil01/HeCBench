include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(image,w,h)
    function launch!()
        @cuda threads=64 blocks=4 to_coef_2d_x!(image,Int32(w),Int32(h))
        @cuda threads=64 blocks=4 to_coef_2d_y!(image,Int32(w),Int32(h))
    end
end
exit(study_main(study_prepare))
