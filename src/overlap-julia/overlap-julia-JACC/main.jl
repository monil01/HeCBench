using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_inc!(i,out,input,n::Int32,inner::Int32)
    idx=Int32(i)-Int32(1)
    if idx<n
        @inbounds for j in Int32(0):inner-Int32(1)
            out[idx+Int32(1)]=(j==Int32(0) ? input[idx+Int32(1)] : out[idx+Int32(1)])+Int32(1)
        end
    end
    nothing
end
function study_prepare(din,dout,n,inner)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    s->JACC.parallel_for(portable_inc!,spec,n,dout[s],din[s],Int32(n),inner)
end
exit(study_main(study_prepare))
