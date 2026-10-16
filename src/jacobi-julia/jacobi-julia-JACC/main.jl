using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function jacobi!(i,j,out,inp,errors,N)
    if i>=2 && i<=N-1 && j>=2 && j<=N-1
        idx=i+(j-1)*N
        @inbounds begin
            value=.25f0*(inp[idx+1]+inp[idx-1]+inp[idx+N]+inp[idx-N]);out[idx]=value
            df=value-inp[idx];error=df*df
        end
        if error>0f0
            JACC.@atomic errors[1]+=error
        end
    end
    return nothing
end
function prepare(a,b,err)
    backend=JACC.get_backend(:cuda)
    spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(128,128),shmem_size=0,sync=false)
    (out,inp,e)->JACC.parallel_for(jacobi!,spec,(STUDY_N,STUDY_N),out,inp,e,Int32(STUDY_N))
end
exit(study_main(prepare))
