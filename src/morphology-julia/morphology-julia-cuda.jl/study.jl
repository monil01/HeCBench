using CUDA
source=replace(read(joinpath(@__DIR__,"main.jl"),String),"exit(main())"=>"")
include_string(Main,source,joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function prepare(a,b)
    function cycle!(img,tmp)
        for op in (Int32(1),Int32(2))
            CUDA.fill!(tmp,UInt8(0))
            @cuda threads=256 blocks=cld(STUDY_W*STUDY_H,256) morphology_kernel!(tmp,img,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op,Int32(1))
            @cuda threads=256 blocks=cld(STUDY_W*STUDY_H,256) morphology_kernel!(img,tmp,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op,Int32(2))
        end
        nothing
    end
    cycle!
end
exit(study_main(prepare))
