using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function glu_kernel!(i,input,output,m,s,n)
    idx=Int32(i)-Int32(1)
    if idx<m*s*n
        outer=idx÷s÷n;mid=(idx÷n)%s;inner=idx%n
        @inbounds begin
            x1=input[outer*Int32(2)*s*n+mid*n+inner+Int32(1)];x2=input[outer*Int32(2)*s*n+(mid+s)*n+inner+Int32(1)]
            output[idx+Int32(1)]=x1*(1f0/(1f0+exp(-x2)))
        end
    end
    nothing
end
function study_prepare(input,output,m,s,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(length(output),256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(glu_kernel!,spec,length(output),input,output,Int32(m),Int32(s),Int32(n))
end
exit(study_main(study_prepare))
