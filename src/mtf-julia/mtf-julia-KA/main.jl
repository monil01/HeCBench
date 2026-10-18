using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function find_symbol_ka!(alphabet,word,rank)
 i=@index(Local,Linear)
 if i==1;rank[1]=Int32(256);end
 @synchronize
 if alphabet[i]==word;rank[1]=Int32(i-1);end
end
const BACKEND=CUDA.CUDABackend()
search(a,w,r)=find_symbol_ka!(BACKEND,256)(a,w,r;ndrange=256)
study((T,n)->KernelAbstractions.zeros(BACKEND,T,n),search)
