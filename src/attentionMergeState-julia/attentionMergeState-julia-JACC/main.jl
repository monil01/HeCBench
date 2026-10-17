using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_merge!(i,output,prefix_output,suffix_output,lse,prefix_lse,suffix_lse,total::Int32,head_size::Int32)
    idx0=Int32(i)-Int32(1)
    if idx0<total
    lse_idx = idx0 ÷ head_size
    p_lse = @inbounds prefix_lse[lse_idx + Int32(1)]
    s_lse = @inbounds suffix_lse[lse_idx + Int32(1)]
    p_lse = isinf(p_lse) ? -Inf32 : p_lse
    s_lse = isinf(s_lse) ? -Inf32 : s_lse
    max_lse = max(p_lse, s_lse)
    p_se = exp(p_lse - max_lse)
    s_se = exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    @inbounds output[idx0 + Int32(1)] = eltype(output)(
        Float32(prefix_output[idx0 + Int32(1)]) * p_scale +
        Float32(suffix_output[idx0 + Int32(1)]) * s_scale)
    if idx0 % head_size == 0
        @inbounds lse[lse_idx + Int32(1)] = log(out_se) + max_lse
    end
    end
    nothing
end
function study_prepare(out,p,s,lse,pl,sl,tokens,heads,size)
    n=tokens*heads*size;backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(portable_merge!,spec,n,out,p,s,lse,pl,sl,Int32(n),Int32(size))
end
exit(study_main(study_prepare))
