using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_gemm!(ix,iy,a,b,c,m::Int32,k::Int32,n::Int32)
    row=Int32(iy)-Int32(1);col=Int32(ix)-Int32(1)
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
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(3,4),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_gemm!,spec,(n,m),a,b,c,Int32(m),Int32(k),Int32(n))
end
exit(study_main(study_prepare))
