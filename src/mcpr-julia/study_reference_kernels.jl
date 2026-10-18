using CUDA
include(joinpath(@__DIR__,"study_arithmetic.jl"))
function aos_cuda!(A,R,P,n,M)
 row=Int32(blockIdx().x-1)*Int32(192)+Int32(threadIdx().x-1);private_body!(row,A,R,P,n,M,false);return
end
function soa_cuda!(A,R,P,n,M)
 row=Int32(blockIdx().x-1)*Int32(192)+Int32(threadIdx().x-1);private_body!(row,A,R,P,n,M,true);return
end
function shared_cuda!(A,R,P,n,M)
 tid=Int32(threadIdx().x-1);row=Int32(blockIdx().x-1)*Int32(96)+tid;s=CUDA.CuDynamicSharedArray(Float64,4032);shared_body!(row,tid,s,A,R,P,n,M);return
end
