using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function find_symbol!(alphabet,word,rank)
 i=threadIdx().x
 if i==1;rank[1]=Int32(256);end
 sync_threads()
 if alphabet[i]==word;rank[1]=Int32(i-1);end
 return
end
search(a,w,r)=CUDA.@cuda threads=256 blocks=1 find_symbol!(a,w,r)
study((T,n)->CUDA.zeros(T,n),search)
