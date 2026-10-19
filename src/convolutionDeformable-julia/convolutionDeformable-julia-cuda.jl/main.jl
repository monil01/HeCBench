using CUDA
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=CuArray(x)
@inline function atomic_add_pixel!(a,i,value)
    CUDA.@atomic a[i]+=value
    return nothing
end
function dcn_im2col!(image,offset,mask,columns,p,n)
    index=(Int32(blockIdx().x)-1)*Int32(512)+Int32(threadIdx().x)-1;step=Int32(512)*Int32(gridDim().x)
    while index<n
        im2col_index!(index,image,offset,mask,columns,p);index+=step
    end
    return nothing
end
function dcn_col2im!(columns,offset,mask,grad_image,p,n)
    index=(Int32(blockIdx().x)-1)*Int32(512)+Int32(threadIdx().x)-1;step=Int32(512)*Int32(gridDim().x)
    while index<n
        col2im_index!(index,columns,offset,mask,grad_image,p);index+=step
    end
    return nothing
end
function dcn_coord!(columns,image,offset,mask,grad_offset,grad_mask,p,n)
    index=(Int32(blockIdx().x)-1)*Int32(512)+Int32(threadIdx().x)-1;step=Int32(512)*Int32(gridDim().x)
    while index<n
        coord_index!(index,columns,image,offset,mask,grad_offset,grad_mask,p);index+=step
    end
    return nothing
end
function launch_pipeline!(w)
    p=w.p;a=w.device;S=p.OH*p.OW;K=p.KH*p.KW
    nc=Int32(p.G*2*K*S);ni=Int32(p.C*K*S);nf=Int32(p.C*S)
    @cuda threads=512 blocks=cld(nc,512) shmem=0 dcn_coord!(a.columns,a.image,a.offset,a.mask,a.go,a.gm,p,nc)
    @cuda threads=512 blocks=cld(ni,512) shmem=0 dcn_col2im!(a.columns,a.offset,a.mask,a.gi,p,ni)
    @cuda threads=512 blocks=cld(nf,512) shmem=0 dcn_im2col!(a.image,a.offset,a.mask,a.columns,p,nf)
end
include("../study_common.jl")
study_main("cuda_jl")
