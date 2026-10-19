using CUDA,KernelAbstractions
const MODEL="ka"
include("../device_common.jl")
function allocate_model(a)
    b=KernelAbstractions.allocate(CUDABackend(),eltype(a),size(a));copyto!(b,a);b
end
@kernel function step_kernel!(a,b,ob,ps,pc,da,omega,nx,ny,t)
    local_id=@index(Local,Linear)
    group_id=@index(Group,Linear)
    tx=Int32(local_id);group0=Int32(group_id)-Int32(1)
    bx=group0%(nx÷128);jj=group0÷(nx÷128);ii=bx*128+tx-1
    vals,v,c=step_values(a,ob,ii,jj,nx,ny,da,omega);store_fields!(b,ii+jj*nx,vals)
    sv=@localmem Float32 (128,)
    sc=@localmem Int32 (128,)
    @inbounds sv[tx]=v
    @inbounds sc[tx]=c
    @synchronize
    if tx==1
        s=0f0;n=Int32(0)
        @inbounds for q=1:128;s+=sv[q];n+=sc[q];end
        idx=bx+jj*(nx÷128)+t*(nx÷128)*ny+17
        @inbounds ps[idx]=s
        @inbounds pc[idx]=n
    end
end
function launch_step(a,b,ob,ps,pc,da,omega,nx,ny,t)
    step_kernel!(CUDABackend(),128)(a,b,ob,ps,pc,da,omega,nx,ny,t;ndrange=Int(nx)*Int(ny))
    nothing
end
include("../study_common.jl")
main()
