using CUDA
function concat_kernel!(a,b,output,sz0,sz2,s1,s2)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
    if i<sz0*sz2*(s1+s2)
        inner=i%sz2;tmp=i÷sz2;mid=tmp%(s1+s2);outer=tmp÷(s1+s2)
        @inbounds output[i+Int32(1)]=mid<s1 ? a[(outer*s1+mid)*sz2+inner+Int32(1)] : b[(outer*s2+mid-s1)*sz2+inner+Int32(1)]
    end
    nothing
end
function study_prepare(a,b,output,sz0,sz2,s1,s2)
    ()->(@cuda threads=256 blocks=cld(length(output),256) concat_kernel!(a,b,output,Int32(sz0),Int32(sz2),Int32(s1),Int32(s2)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
