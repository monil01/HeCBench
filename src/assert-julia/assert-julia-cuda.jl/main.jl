using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function test_kernel(N)
 gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 CUDA.@cuassert gid<N
 return
end
function perf_kernel(extra::Bool)
 gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 if extra;CUDA.@cuassert gid<=blockDim().x*gridDim().x;end
 s=Int32(0)
 for n=Int32(1):gid
  s+=Int32(1);CUDA.@cuassert s<=gid
 end
 return
end
assertion_study(N->(@cuda threads=32 blocks=2 test_kernel(N)), k->(@cuda threads=256 blocks=1000 perf_kernel(k==1)))
