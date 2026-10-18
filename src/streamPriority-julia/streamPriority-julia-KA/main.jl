using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
function launch(k,pref,w,gin,a,gout,n,batch)
 ni=Int32(n);bi=Int32(batch);nd=cld(n,256)*256
 if k==1;forward_ka!(BACKEND,256)(pref,a,ni;ndrange=nd)
 elseif k==2;backward_ka!(BACKEND,256)(a,gin,gout,ni;ndrange=nd)
 elseif k==3;update_ka!(BACKEND,256)(w,gout,ni;ndrange=nd)
 else;prefetch_ka!(BACKEND,256)(pref,ni,bi;ndrange=nd)
 end
end
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
