# Correctness-only diagnostic of the byte-preserved historical kernel.
using CUDA
using Printf
source=replace(read(joinpath(@__DIR__,"morphology-julia-cuda.jl","legacy_main.jl"),String),"exit(main(ARGS))"=>"")
include_string(Main,source,joinpath(@__DIR__,"morphology-julia-cuda.jl","legacy_main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
function check_legacy()
    for (case,input) in enumerate(fixture_images())
        expected=copy(input);tmp=similar(expected)
        for _ in 1:10,op in (1,2)
            study_axis_cpu!(tmp,expected,STUDY_W,STUDY_H,STUDY_SEL,op,1)
            study_axis_cpu!(expected,tmp,STUDY_W,STUDY_H,STUDY_SEL,op,2)
        end
        a=CuArray(input);b=similar(a)
        for _ in 1:10,op in (Int32(1),Int32(2))
            @cuda threads=256 blocks=cld(STUDY_W*STUDY_H,256) morphology_kernel!(b,a,Int32(STUDY_W),Int32(STUDY_H),Int32(STUDY_SEL),Int32(STUDY_SEL),op)
            copyto!(a,b)
        end
        CUDA.synchronize();got=Array(a);mismatches=count(got.!=expected);maxabs=maximum(abs.(Int16.(got).-Int16.(expected)))
        @printf("LEGACY_REFERENCE_CHECK {\"case\":%d,\"correct\":%s,\"mismatches\":%d,\"max_abs_error\":%d,\"checked_values\":%d}\n",case,string(mismatches==0),mismatches,maxabs,length(got))
    end
end
check_legacy()
