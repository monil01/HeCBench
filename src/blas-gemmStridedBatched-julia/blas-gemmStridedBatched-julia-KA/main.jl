using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
# Genuine custom Float64 rowdot kernel; native/reference use opaque vendor GEMM.
@kernel function batched_ka!(A,B,C,m,k,upper,batch)
    index=(@index(Global,Linear))-1
    if index<m*batch
        row=index%m;b=index÷m;value=0.0
        for col in 0:k-1
            @inbounds value+=A[b*upper*upper+row+col*upper+1]*B[b*upper+col+1]
        end
        @inbounds C[b*upper+row+1]=value
    end
end
batched_study((A,B,C,m,k,upper,batch)->()->batched_ka!(CUDABackend(),256)(A,B,C,m,k,upper,batch;ndrange=m*batch))
