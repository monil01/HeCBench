using CUDA,StaticArrays
include(joinpath(@__DIR__,"..","semblance_device.jl"))
function init_c_kernel(c,inc,c0)
    i=blockIdx().x
    init_c_at(i,c,inc,c0);nothing
end
function init_half_kernel(s,gx,gy,sx,sy,h)
    i=blockIdx().x
    init_half_at(i,s,gx,gy,sx,sy,h);nothing
end
function semblance_kernel(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    semblance_at(i,h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns);nothing
end
function redux_kernel(num,stt,ctr,str,stk,nc,cdp,ns)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    redux_at(i,num,stt,ctr,str,stk,nc,cdp,ns);nothing
end
launch_c(n,c,inc,c0)=(@cuda threads=1 blocks=n init_c_kernel(c,inc,c0);nothing)
launch_half(n,s,gx,gy,sx,sy,h)=(@cuda threads=1 blocks=n init_half_kernel(s,gx,gy,sx,sy,h);nothing)
launch_scan(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)=(@cuda threads=128 blocks=cld(ns*nc,128) semblance_kernel(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns);nothing)
launch_redux(num,stt,ctr,str,stk,nc,cdp,ns)=(@cuda threads=128 blocks=cld(ns,128) redux_kernel(num,stt,ctr,str,stk,nc,cdp,ns);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
