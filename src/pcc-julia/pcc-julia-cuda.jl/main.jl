using CUDA
include("../extract_device.jl")
function single_kernel(C,U,n)
 i=Int((blockIdx().x-1)*blockDim().x+threadIdx().x);extract_site!(C,U,i,n,n);nothing
end
function multi_kernel(C,U,n1,n)
 i=Int((blockIdx().x-1)*blockDim().x+threadIdx().x);extract_site!(C,U,i,n1,n);nothing
end
function extract(c,ch)
 if c.avail==0
  @cuda threads=1024 blocks=cld(ch.n*ch.n,1024) single_kernel(ch.C,ch.U,ch.n)
 else
  @cuda threads=1024 blocks=cld(ch.n1*ch.n,1024) multi_kernel(ch.C,ch.U,ch.n1,ch.n)
 end
 nothing
end
include("../study_common.jl")
pcc_study(extract)
