using CUDA
using KernelAbstractions
using Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function jacobi!(out,inp,errors,N)
    i,j=@index(Global,NTuple)
    if i>=2 && i<=N-1 && j>=2 && j<=N-1
        idx=i+(j-1)*N
        @inbounds begin
            value=.25f0*(inp[idx+1]+inp[idx-1]+inp[idx+N]+inp[idx-N]);out[idx]=value
            df=value-inp[idx];error=df*df
        end
        if error>0f0
            Atomix.@atomic errors[1]+=error
        end
    end
end
function prepare(a,b,err)
    kernel=jacobi!(KernelAbstractions.get_backend(a),(16,16))
    (out,inp,e)->kernel(out,inp,e,Int32(STUDY_N);ndrange=(STUDY_N,STUDY_N))
end
exit(study_main(prepare))
