using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function concat_kernel!(idx,a,b,output,sz0,sz2,s1,s2)
    i=Int32(idx)-Int32(1)
    if i<sz0*sz2*(s1+s2)
        inner=i%sz2;tmp=i÷sz2;mid=tmp%(s1+s2);outer=tmp÷(s1+s2)
        @inbounds output[i+Int32(1)]=mid<s1 ? a[(outer*s1+mid)*sz2+inner+Int32(1)] : b[(outer*s2+mid-s1)*sz2+inner+Int32(1)]
    end
    nothing
end
function study_prepare(a,b,output,sz0,sz2,s1,s2)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(length(output),256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(concat_kernel!,spec,length(output),a,b,output,Int32(sz0),Int32(sz2),Int32(s1),Int32(s2))
end
exit(study_main(study_prepare))
