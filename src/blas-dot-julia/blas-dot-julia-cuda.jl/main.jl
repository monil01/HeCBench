using CUDA
# Actual native cublasDotEx semantics with device scalar result and explicit execution type.
function study_prepare(a,b,result,n)
    T=eltype(a);Acc=T==Float64 ? Float64 : Float32
    ()->CUDA.CUBLAS.cublasDotEx(CUDA.CUBLAS.handle(),n,a,T,1,b,T,1,result,T,Acc)
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
