include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"exit\(main\(\)\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare()
    function launch!(mode,img,h1,h2,tn,bn,mp)
        n=Int32(length(img))
        if mode==1
            @cuda threads=256 blocks=256 merge_kernel!(n,img,h1,h2,tn,bn)
        else
            @cuda threads=256 blocks=256 find_moving_pixels_kernel!(n,img,h1,h2,tn,mp)
            @cuda threads=256 blocks=256 update_background_kernel!(n,img,mp,bn)
            @cuda threads=256 blocks=256 update_threshold_kernel!(n,img,mp,bn,tn)
        end
    end
end
exit(study_main(study_prepare))
