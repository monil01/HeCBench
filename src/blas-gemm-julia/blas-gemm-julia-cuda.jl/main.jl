using CUDA
# Repaired reference: actual original BLAS GEMM semantics, not historical smoke output.
function study_prepare(a,b,c,m,k,n)
    T=eltype(c);am=reshape(a,k,m);bm=reshape(b,n,k);cm=reshape(c,n,m)
    ()->CUDA.CUBLAS.gemm!('N','N',T(2),bm,am,T(.5),cm)
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"))
    exit(study_main(study_prepare))
end
