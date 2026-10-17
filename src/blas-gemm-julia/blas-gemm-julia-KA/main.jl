using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_gemm!(a,b,c,m::Int32,k::Int32,n::Int32)
    ix,iy=@index(Global,NTuple);row=Int32(iy)-Int32(1);col=Int32(ix)-Int32(1)
    if row<m && col<n
        T=eltype(c);s=zero(T)
        @inbounds for q in Int32(0):k-Int32(1)
            s=T(s+T(a[row*k+q+Int32(1)]*b[q*n+col+Int32(1)]))
        end
        idx=row*n+col+Int32(1)
        @inbounds c[idx]=T(T(2)*s+T(.5)*c[idx])
    end
    nothing
end
function study_prepare(a,b,c,m,k,n)
    kernel=portable_gemm!(CUDABackend(),(16,16));()->kernel(a,b,c,Int32(m),Int32(k),Int32(n);ndrange=(n,m))
end
exit(study_main(study_prepare))
