# Own KA device entries/dispatch; CUDA backend intrinsics explicit in shared SGD body.
using CUDA, KernelAbstractions
const MODEL="ka"
include("../study_device.jl")
@kernel function kernel_rng_sized!(seed,state,n)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_rng_sized!(gid,seed,state,n)
 nothing
end
@kernel function kernel_rng_unsized!(seed,state)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_rng_unsized!(gid,seed,state)
 nothing
end
@kernel function kernel_features!(stride,state,state_size,features,array_size,k,scale)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_features!(gid,stride,state,state_size,features,array_size,k,scale)
 nothing
end
@kernel function kernel_transform!(stride,input,output,n)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_transform!(gid,stride,input,output,n)
 nothing
end
@kernel function kernel_sgd!(R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_sgd!(gid,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 nothing
end
function dispatch!(::Val{:rng_sized},s,blocks,threads,seed,state,n)
 s.launches[1]+=1
 kernel_rng_sized!(CUDABackend(),threads)(seed,state,n;ndrange=blocks*threads)
 nothing
end
function dispatch!(::Val{:rng_unsized},s,blocks,threads,seed,state)
 s.launches[1]+=1
 kernel_rng_unsized!(CUDABackend(),threads)(seed,state;ndrange=blocks*threads)
 nothing
end
function dispatch!(::Val{:features},s,blocks,threads,stride,state,state_size,features,array_size,k,scale)
 s.launches[4]+=1
 kernel_features!(CUDABackend(),threads)(stride,state,state_size,features,array_size,k,scale;ndrange=blocks*threads)
 nothing
end
function dispatch!(::Val{:transform},s,blocks,threads,stride,input,output,n)
 s.launches[3]+=1
 kernel_transform!(CUDABackend(),threads)(stride,input,output,n;ndrange=blocks*threads)
 nothing
end
function dispatch!(::Val{:sgd},s,blocks,threads,R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace)
 s.launches[2]+=1
 kernel_sgd!(CUDABackend(),threads)(R,nnz,p,q,state,rates,u_seg,v_seg,k,num_iters,current_iter,update_count_per_block,updates,vector_size,lambda_p,lambda_q,u_grid,v_grid,u_id,v_id,trace;ndrange=blocks*threads)
 nothing
end
include("../study_common.jl")
main()
