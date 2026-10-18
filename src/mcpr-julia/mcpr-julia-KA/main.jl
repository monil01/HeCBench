using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
@kernel function aos_ka!(A,R,P,n,M)
 row=@index(Global,Linear);private_body!(Int32(row-1),A,R,P,n,M,false)
end
@kernel function soa_ka!(A,R,P,n,M)
 row=@index(Global,Linear);private_body!(Int32(row-1),A,R,P,n,M,true)
end
@kernel function shared_ka!(A,R,P,n,M)
 tid=@index(Local,Linear);bid=@index(Group,Linear);s=@localmem Float64 (4032,);row=Int32((bid-1)*96+tid-1);shared_body!(row,Int32(tid-1),s,A,R,P,n,M)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
const AOS=aos_ka!(BACKEND,192)
const SOA=soa_ka!(BACKEND,192)
const SHARED=shared_ka!(BACKEND,96)
function launch(variant,A,R,P,n,M)
 threads=variant==3 ? 96 : 192;kernel=variant==1 ? AOS : (variant==2 ? SOA : SHARED);kernel(A,R,P,n,M;ndrange=cld(Int(n),threads)*threads);return
end
study(launch,(T,n)->KernelAbstractions.allocate(BACKEND,T,n))
