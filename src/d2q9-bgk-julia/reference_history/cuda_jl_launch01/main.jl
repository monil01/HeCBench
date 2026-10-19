using CUDA
const MODEL="cuda_jl"
include("../device_common.jl")
allocate_model(a)=CuArray(a)
function step_kernel(a,b,ob,ps,pc,da,omega,nx,ny,t)
    tx=Int32(threadIdx().x);bx=Int32(blockIdx().x)-1;jj=Int32(blockIdx().y)-1;ii=bx*128+tx-1
    vals,v,c=step_values(a,ob,ii,jj,nx,ny,da,omega);store_fields!(b,ii+jj*nx,vals)
    sv=@cuStaticSharedMem(Float32,128);sc=@cuStaticSharedMem(Int32,128)
    @inbounds sv[tx]=v;@inbounds sc[tx]=c;sync_threads()
    if tx==1
        s=0f0;n=Int32(0)
        @inbounds for q=1:128;s+=sv[q];n+=sc[q];end
        idx=bx+jj*(nx÷128)+t*(nx÷128)*ny+17
        @inbounds ps[idx]=s;@inbounds pc[idx]=n
    end
    return nothing
end
function launch_step(a,b,ob,ps,pc,da,omega,nx,ny,t)
    @cuda threads=(128,1) blocks=(nx÷128,ny) step_kernel(a,b,ob,ps,pc,da,omega,nx,ny,t)
    nothing
end
include("../study_common.jl")
main()
