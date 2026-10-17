using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function batched_jacc!(global_index,A,B,C,m,k,upper,batch)
    index=global_index-1
    if index<m*batch
        row=index%m;b=index÷m;value=0.0
        for col in 0:k-1
            @inbounds value+=A[b*upper*upper+row+col*upper+1]*B[b*upper+col+1]
        end
        @inbounds C[b*upper+row+1]=value
    end
    return
end
batched_study((A,B,C,m,k,upper,batch)->()->JACC.parallel_for(JACC.launch_spec(;threads=256,blocks=cld(m*batch,256),shmem_size=0,sync=false),m*batch,batched_jacc!,A,B,C,m,k,upper,batch))
