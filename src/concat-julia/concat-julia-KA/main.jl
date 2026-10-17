using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function concat_kernel!(a,b,output,sz0,sz2,s1,s2)
    i=Int32(@index(Global,Linear))-Int32(1)
    if i<sz0*sz2*(s1+s2)
        inner=i%sz2;tmp=i÷sz2;mid=tmp%(s1+s2);outer=tmp÷(s1+s2)
        @inbounds output[i+Int32(1)]=mid<s1 ? a[(outer*s1+mid)*sz2+inner+Int32(1)] : b[(outer*s2+mid-s1)*sz2+inner+Int32(1)]
    end
    nothing
end
function study_prepare(a,b,output,sz0,sz2,s1,s2)
    k=concat_kernel!(CUDABackend(),256)
    ()->k(a,b,output,Int32(sz0),Int32(sz2),Int32(s1),Int32(s2);ndrange=length(output))
end
exit(study_main(study_prepare))
