using CUDA
import Atomix
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function path_kernel(s,p,n,steps,K,S0,r,sigma,dt,put)
 g=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 paths_body!(s,p,n,steps,K,S0,r,sigma,dt,put,g)
 nothing
end
function prepare_kernel(p,f,v,n,K,put)
 ints=CUDA.CuStaticSharedArray(Int32,514);mem=CUDA.CuStaticSharedArray(Float64,16)
 prepare_body!(p,f,v,n,K,put,Int32(blockIdx().x-1),Int32(threadIdx().x-1),ints,mem,CUDA.sync_threads)
 nothing
end
function partial_beta_kernel(p,f,v,t,n,step,K,put)
 mem=CUDA.CuStaticSharedArray(Float64,15)
 partial_beta_body!(p,f,v,t,n,step,K,put,Int32(blockIdx().x-1),Int32(threadIdx().x-1),mem,CUDA.sync_threads)
 nothing
end
function final_beta_kernel(f,t,step)
 mem=CUDA.CuStaticSharedArray(Float64,3)
 final_beta_body!(f,t,step,Int32(threadIdx().x-1),mem,CUDA.sync_threads)
 nothing
end
function cash_kernel(p,f,t,n,step,K,put,discount)
 g=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);stride=Int32(gridDim().x*blockDim().x)
 cash_body!(p,f,t,n,step,K,put,discount,g,stride)
 nothing
end
function partial_sum_kernel(p,t,n)
 mem=CUDA.CuStaticSharedArray(Float64,1)
 partial_sum_body!(p,t,n,Int32(blockIdx().x-1),Int32(threadIdx().x-1),mem,CUDA.sync_threads)
 nothing
end
function final_sum_kernel(t,n,blocks,discount)
 mem=CUDA.CuStaticSharedArray(Float64,1)
 final_sum_body!(t,n,blocks,discount,Int32(threadIdx().x-1),mem,CUDA.sync_threads)
 nothing
end
function launch(kind,args...;threads,blocks)
 if kind==1;@cuda threads=threads blocks=blocks path_kernel(args...)
 elseif kind==2;@cuda threads=threads blocks=blocks prepare_kernel(args...)
 elseif kind==3;@cuda threads=threads blocks=blocks partial_beta_kernel(args...)
 elseif kind==4;@cuda threads=threads blocks=blocks final_beta_kernel(args...)
 elseif kind==5;@cuda threads=threads blocks=blocks cash_kernel(args...)
 elseif kind==6;@cuda threads=threads blocks=blocks partial_sum_kernel(args...)
 else;@cuda threads=threads blocks=blocks final_sum_kernel(args...)
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch)
