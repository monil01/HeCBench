using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function glu_kernel!(input,output,m,s,n)
    idx=Int32(@index(Global,Linear))-Int32(1)
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
    k=glu_kernel!(CUDABackend(),256)
    ()->k(input,output,Int32(m),Int32(s),Int32(n);ndrange=length(output))
end
exit(study_main(study_prepare))
