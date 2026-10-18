using CUDA,JACC
include("../extract_device.jl")
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=1024,blocks=1,shmem_size=0,sync=false)
function single_kernel!(i,C,U,n)
 extract_site!(C,U,i,n,n);nothing
end
function multi_kernel!(i,C,U,n1,n)
 extract_site!(C,U,i,n1,n);nothing
end
function extract(c,ch)
 if c.avail==0;JACC.parallel_for(single_kernel!,SPEC,ch.n*ch.n,ch.C,ch.U,ch.n)
 else;JACC.parallel_for(multi_kernel!,SPEC,ch.n1*ch.n,ch.C,ch.U,ch.n1,ch.n);end
 nothing
end
include("../study_common.jl")
pcc_study(extract)
