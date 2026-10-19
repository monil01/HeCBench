using CUDA, KernelAbstractions
import Atomix
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=begin
 a=KernelAbstractions.allocate(CUDABackend(),eltype(x),size(x)...);copyto!(a,x);a
end
@inline function atomic_add_pixel!(a,i,value)
    Atomix.@atomic a[i]+=value
end
@kernel function dcn_im2col!(image,offset,mask,columns,p,n)
    index=Int32(@index(Global,Linear))-1;step=cld(n,Int32(512))*Int32(512)
    while index<n
        im2col_index!(index,image,offset,mask,columns,p);index+=step
    end
end
@kernel function dcn_col2im!(columns,offset,mask,grad_image,p,n)
    index=Int32(@index(Global,Linear))-1;step=cld(n,Int32(512))*Int32(512)
    while index<n
        col2im_index!(index,columns,offset,mask,grad_image,p);index+=step
    end
end
@kernel function dcn_coord!(columns,image,offset,mask,grad_offset,grad_mask,p,n)
    index=Int32(@index(Global,Linear))-1;step=cld(n,Int32(512))*Int32(512)
    while index<n
        coord_index!(index,columns,image,offset,mask,grad_offset,grad_mask,p);index+=step
    end
end
function launch_pipeline!(w)
    p=w.p;a=w.device;S=p.OH*p.OW;K=p.KH*p.KW
    nc=Int32(p.G*2*K*S);ni=Int32(p.C*K*S);nf=Int32(p.C*S)
    dcn_coord!(CUDABackend(),512)(a.columns,a.image,a.offset,a.mask,a.go,a.gm,p,nc;ndrange=cld(nc,512)*512)
    dcn_col2im!(CUDABackend(),512)(a.columns,a.offset,a.mask,a.gi,p,ni;ndrange=cld(ni,512)*512)
    dcn_im2col!(CUDABackend(),512)(a.image,a.offset,a.mask,a.columns,p,nf;ndrange=cld(nf,512)*512)
end
include("../study_common.jl")
study_main("ka")
