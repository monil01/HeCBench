using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function test_kernel(index,N)
 gid=Int32(index)-Int32(1)
 CUDA.@cuassert gid<N
 return
end
function perf_kernel(index,extra::Bool,total::Int32)
 gid=Int32(index)-Int32(1)
 if extra;CUDA.@cuassert gid<=total;end
 s=Int32(0)
 for n=Int32(1):gid
  s+=Int32(1);CUDA.@cuassert s<=gid
 end
 return
end
const test_spec=JACC.launch_spec(;threads=32,blocks=2,shmem_size=0,sync=false)
const perf_spec=JACC.launch_spec(;threads=256,blocks=1000,shmem_size=0,sync=false)
assertion_study(N->JACC.parallel_for(test_spec,64,test_kernel,N),k->JACC.parallel_for(perf_spec,256000,perf_kernel,k==1,Int32(256000)))
