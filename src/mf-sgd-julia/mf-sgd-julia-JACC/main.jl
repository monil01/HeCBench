# Own JACC callbacks/dispatch; source cached loads/warp helpers explicitly CUDA-specific.
using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../study_device.jl")
function kernel_rng_sized!(index,seed,state,n)
 gid=Int32(index)-Int32(1)
 body_rng_sized!(gid,seed,state,n)
 nothing
end
function kernel_rng_unsized!(index,seed,state)
 gid=Int32(index)-Int32(1)
 body_rng_unsized!(gid,seed,state)
 nothing
end
function kernel_features!(index,stride,state,state_size,features,array_size,k,scale)
 gid=Int32(index)-Int32(1)
 body_features!(gid,stride,state,state_size,features,array_size,k,scale)
 nothing
end
function kernel_transform!(index,stride,input,output,n)
 gid=Int32(index)-Int32(1)
 body_transform!(gid,stride,input,output,n)
 nothing
end
function kernel_sgd!(index,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 gid=Int32(index)-Int32(1)
 body_sgd!(gid,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 nothing
end
function dispatch!(::Val{:rng_sized},s,blocks,threads,seed,state,n)
 s.launches[1]+=1
 spec=JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*threads,kernel_rng_sized!,seed,state,n)
 nothing
end
function dispatch!(::Val{:rng_unsized},s,blocks,threads,seed,state)
 s.launches[1]+=1
 spec=JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*threads,kernel_rng_unsized!,seed,state)
 nothing
end
function dispatch!(::Val{:features},s,blocks,threads,stride,state,state_size,features,array_size,k,scale)
 s.launches[4]+=1
 spec=JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*threads,kernel_features!,stride,state,state_size,features,array_size,k,scale)
 nothing
end
function dispatch!(::Val{:transform},s,blocks,threads,stride,input,output,n)
 s.launches[3]+=1
 spec=JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*threads,kernel_transform!,stride,input,output,n)
 nothing
end
function dispatch!(::Val{:sgd},s,blocks,threads,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 s.launches[2]+=1
 spec=JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*threads,kernel_sgd!,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 nothing
end
include("../study_common.jl")
main()
