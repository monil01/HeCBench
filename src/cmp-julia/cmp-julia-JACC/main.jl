using CUDA,StaticArrays,JACC
include(joinpath(@__DIR__,"..","semblance_device.jl"))
@inline init_c_kernel(i,c,inc,c0)=init_c_at(i,c,inc,c0)
@inline init_half_kernel(i,s,gx,gy,sx,sy,h)=init_half_at(i,s,gx,gy,sx,sy,h)
@inline semblance_kernel(i,h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)=semblance_at(i,h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)
@inline redux_kernel(i,num,stt,ctr,str,stk,nc,cdp,ns)=redux_at(i,num,stt,ctr,str,stk,nc,cdp,ns)
# Explicit complete dimensions: LaunchSpec is mutable and must not cache a
# small warmup's inferred grid for another case. All creation is outside ROI.
const init_specs=Dict(n=>JACC.launch_spec(;threads=1,blocks=n,shmem_size=0,sync=false) for n in (5,9,16,52))
const scan_specs=Dict(n=>JACC.launch_spec(;threads=128,blocks=cld(n,128),shmem_size=0,sync=false) for n in (1285,9225,257,1025))
launch_c(n,c,inc,c0)=(JACC.parallel_for(init_specs[n],n,init_c_kernel,c,inc,c0);nothing)
launch_half(n,s,gx,gy,sx,sy,h)=(JACC.parallel_for(init_specs[n],n,init_half_kernel,s,gx,gy,sx,sy,h);nothing)
launch_scan(h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns)=(JACC.parallel_for(scan_specs[ns*nc],ns*nc,semblance_kernel,h,c,samples,num,stt,a,b,idt,dt,tau,w,nc,ns);nothing)
launch_redux(num,stt,ctr,str,stk,nc,cdp,ns)=(JACC.parallel_for(scan_specs[ns],ns,redux_kernel,num,stt,ctr,str,stk,nc,cdp,ns);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
