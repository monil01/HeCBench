using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(k,pref,w,gin,a,gout,n,batch)
 ni=Int32(n);bi=Int32(batch);blocks=cld(n,256);nd=blocks*256;spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 if k==1;JACC.parallel_for(spec,nd,forward_jacc!,pref,a,ni)
 elseif k==2;JACC.parallel_for(spec,nd,backward_jacc!,a,gin,gout,ni)
 elseif k==3;JACC.parallel_for(spec,nd,update_jacc!,w,gout,ni)
 else;JACC.parallel_for(spec,nd,prefetch_jacc!,pref,ni,bi)
 end
end
study((T,n)->JACC.zeros(T,n),launch)
