using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(k,bs,x,g,o,dx,B,C,H,W)
 dims=(UInt64(B),UInt64(C),UInt64(H),UInt64(W));S=B*C*H*W
 k==1&&return JACC.parallel_for(JACC.launch_spec(threads=bs,blocks=cld(S,bs),shmem_size=0,sync=false),cld(S,bs)*bs,up1!,x,o,dims...)
 k==2&&return JACC.parallel_for(JACC.launch_spec(threads=(bs,bs,1),blocks=(cld(W,bs),cld(H,bs),B*C),shmem_size=0,sync=false),(cld(W,bs)*bs,cld(H,bs)*bs,B*C),up2!,x,o,dims...)
 k==3&&return JACC.parallel_for(JACC.launch_spec(threads=bs,blocks=cld(S,bs),shmem_size=0,sync=false),cld(S,bs)*bs,up3!,g,dx,dims...)
 k==4&&return JACC.parallel_for(JACC.launch_spec(threads=(bs,bs,1),blocks=(cld(W,bs),cld(H,bs),B*C),shmem_size=0,sync=false),(cld(W,bs)*bs,cld(H,bs)*bs,B*C),up4!,g,dx,dims...)
end
study((T,n)->JACC.zeros(T,n),launch)
