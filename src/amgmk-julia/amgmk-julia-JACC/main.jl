using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
function relax_kernel!(i,data,idx,cols,u,f,n::Int32)
 amg_relax(Int32(i)-Int32(1),n,data,idx,cols,u,f)
 nothing
end
function observe_kernel!(i,p,n::Int32)
 @inbounds p[1]=n;p[2]=Int32(NX);p[3]=Int32(NY);p[4]=Int32(NZ);p[5]=Int32(R)
 nothing
end
const RELAX=JACC.launch_spec(threads=256,blocks=cld(N,256),shmem_size=0,sync=false)
const OBSERVE=JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false)
amg_study((args...)->JACC.parallel_for(RELAX,cld(N,256)*256,relax_kernel!,args...),(p,n)->JACC.parallel_for(OBSERVE,1,observe_kernel!,p,n),"jacc";allocate=JACC.array)
