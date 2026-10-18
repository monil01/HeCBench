using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATES=(JACC.zeros(Float32,1),JACC.zeros(Float32,4352),JACC.zeros(Float32,4368),JACC.zeros(Float32,4416),JACC.zeros(Float32,4416),JACC.zeros(Float32,4160),JACC.zeros(Float32,8192),JACC.zeros(Float32,8192),JACC.zeros(Float32,8192),JACC.zeros(Float32,5120))
function launch(k,a,b,o,idx)
 k==1&&return JACC.parallel_for(JACC.launch_spec(threads=(128,1),blocks=(1,1),shmem_size=0,sync=false),(128,1),match1!,a,b,o,idx,TEMPLATES[1])
 k==2&&return JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(8,1),shmem_size=17408,sync=false),(128,16),match2!,a,b,o,idx,TEMPLATES[2])
 k==3&&return JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(8,1),shmem_size=17472,sync=false),(128,16),match3!,a,b,o,idx,TEMPLATES[3])
 k==4&&return JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(8,1),shmem_size=17664,sync=false),(128,16),match4!,a,b,o,idx,TEMPLATES[4])
 k==5&&return JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(8,1),shmem_size=17664,sync=false),(128,16),match5!,a,b,o,idx,TEMPLATES[5])
 k==6&&return JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(8,1),shmem_size=16640,sync=false),(128,16),match6!,a,b,o,idx,TEMPLATES[6])
 k==7&&return JACC.parallel_for(JACC.launch_spec(threads=(32,8),blocks=(4,1),shmem_size=32768,sync=false),(128,8),match7!,a,b,o,idx,TEMPLATES[7])
 k==8&&return JACC.parallel_for(JACC.launch_spec(threads=(32,8),blocks=(4,1),shmem_size=32768,sync=false),(128,8),match8!,a,b,o,idx,TEMPLATES[8])
 k==9&&return JACC.parallel_for(JACC.launch_spec(threads=(32,4),blocks=(4,1),shmem_size=32768,sync=false),(128,4),match9!,a,b,o,idx,TEMPLATES[9])
 k==10&&return JACC.parallel_for(JACC.launch_spec(threads=(32,8),blocks=(4,1),shmem_size=20480,sync=false),(128,8),match10!,a,b,o,idx,TEMPLATES[10])
end
study((T,n)->JACC.zeros(T,n),launch)
