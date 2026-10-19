using CUDA,StaticArrays,KernelAbstractions
include(joinpath(@__DIR__,"..","semblance_device.jl"))
@kernel function init_c_kernel(c,inc,c0)
    i=@index(Global,Linear)
    init_c_at(i,c,inc,c0);nothing
end
@kernel function init_half_kernel(s,gx,gy,sx,sy,h)
    i=@index(Global,Linear)
    init_half_at(i,s,gx,gy,sx,sy,h);nothing
end
@kernel function semblance_kernel(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)
    i=@index(Global,Linear)
    semblance_at(i,h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns);nothing
end
@kernel function redux_kernel(num,stt,ctr,str,stk,nc,cdp,ns)
    i=@index(Global,Linear)
    redux_at(i,num,stt,ctr,str,stk,nc,cdp,ns);nothing
end
launch_c(n,c,inc,c0)=(init_c_kernel(CUDA.CUDABackend(),1)(c,inc,c0;ndrange=n);nothing)
launch_half(n,s,gx,gy,sx,sy,h)=(init_half_kernel(CUDA.CUDABackend(),1)(s,gx,gy,sx,sy,h;ndrange=n);nothing)
launch_scan(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)=(semblance_kernel(CUDA.CUDABackend(),128)(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns;ndrange=ns*nc);nothing)
launch_redux(num,stt,ctr,str,stk,nc,cdp,ns)=(redux_kernel(CUDA.CUDABackend(),128)(num,stt,ctr,str,stk,nc,cdp,ns;ndrange=ns);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
