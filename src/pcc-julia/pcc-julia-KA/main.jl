using CUDA,KernelAbstractions
include("../extract_device.jl")
@kernel unsafe_indices=true function single_kernel!(C,U,n)
 extract_site!(C,U,Int(@index(Global,Linear)),n,n)
end
@kernel unsafe_indices=true function multi_kernel!(C,U,n1,n)
 extract_site!(C,U,Int(@index(Global,Linear)),n1,n)
end
const SINGLE=single_kernel!(CUDABackend(),1024);const MULTI=multi_kernel!(CUDABackend(),1024)
function extract(c,ch)
 if c.avail==0;SINGLE(ch.C,ch.U,ch.n;ndrange=cld(ch.n*ch.n,1024)*1024)
 else;MULTI(ch.C,ch.U,ch.n1,ch.n;ndrange=cld(ch.n1*ch.n,1024)*1024);end
 nothing
end
include("../study_common.jl")
pcc_study(extract)
