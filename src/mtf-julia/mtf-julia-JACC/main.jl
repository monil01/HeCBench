using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function find_symbol_jacc!(i,alphabet,word,rank)
 if i==1;rank[1]=Int32(256);end
 JACC.sync_workgroup()
 if alphabet[i]==word;rank[1]=Int32(i-1);end
 return
end
search(a,w,r)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=1,shmem_size=0,sync=false),256,find_symbol_jacc!,a,w,r)
study((T,n)->JACC.zeros(T,n),search)
