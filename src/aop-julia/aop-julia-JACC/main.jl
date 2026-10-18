using CUDA
import JACC, Atomix
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
const TEMPLATES=Dict(n=>JACC.array(zeros(Float64,n)) for n in [1,3,15,273])
function path_kernel(i,s,p,n,steps,K,S0,r,sigma,dt,put)
 paths_body!(s,p,n,steps,K,S0,r,sigma,dt,put,Int32(i-1))
 nothing
end
function prepare_kernel(i,p,f,v,n,K,put,template)
 arena=JACC.shared(template);mem=view(arena,1:16);ints=view(reinterpret(Int32,arena),33:546)
 prepare_body!(p,f,v,n,K,put,Int32((i-1)÷256),Int32((i-1)%256),ints,mem,JACC.sync_workgroup)
 nothing
end
function partial_beta_kernel(i,p,f,v,t,n,step,K,put,template)
 mem=JACC.shared(template)
 partial_beta_body!(p,f,v,t,n,step,K,put,Int32((i-1)÷128),Int32((i-1)%128),mem,JACC.sync_workgroup)
 nothing
end
function final_beta_kernel(i,f,t,step,template)
 mem=JACC.shared(template)
 final_beta_body!(f,t,step,Int32((i-1)%128),mem,JACC.sync_workgroup)
 nothing
end
function cash_kernel(i,p,f,t,n,step,K,put,discount,stride)
 cash_body!(p,f,t,n,step,K,put,discount,Int32(i-1),stride)
 nothing
end
function partial_sum_kernel(i,p,t,n,template)
 mem=JACC.shared(template)
 partial_sum_body!(p,t,n,Int32((i-1)÷128),Int32((i-1)%128),mem,JACC.sync_workgroup)
 nothing
end
function final_sum_kernel(i,t,n,blocks,discount,template)
 mem=JACC.shared(template)
 final_sum_body!(t,n,blocks,discount,Int32((i-1)%128),mem,JACC.sync_workgroup)
 nothing
end
function launch(kind,args...;threads,blocks)
 elements=kind==2 ? 273 : kind==3 ? 15 : kind==4 ? 3 : kind in (6,7) ? 1 : 0
 spec=JACC.launch_spec(;threads=threads,blocks=blocks,shmem_size=8*elements,sync=false)
 n=threads*blocks
 if kind==1;JACC.parallel_for(spec,n,path_kernel,args...)
 elseif kind==2;JACC.parallel_for(spec,n,prepare_kernel,args...,TEMPLATES[elements])
 elseif kind==3;JACC.parallel_for(spec,n,partial_beta_kernel,args...,TEMPLATES[elements])
 elseif kind==4;JACC.parallel_for(spec,n,final_beta_kernel,args...,TEMPLATES[elements])
 elseif kind==5;JACC.parallel_for(spec,n,cash_kernel,args...,Int32(n))
 elseif kind==6;JACC.parallel_for(spec,n,partial_sum_kernel,args...,TEMPLATES[elements])
 else;JACC.parallel_for(spec,n,final_sum_kernel,args...,TEMPLATES[elements])
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
