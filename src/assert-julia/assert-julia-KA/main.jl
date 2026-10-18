using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function test_kernel(N)
 gid=Int32(@index(Global,Linear))-Int32(1)
 CUDA.@cuassert gid<N
end
@kernel function perf_kernel(extra::Bool)
 gid=Int32(@index(Global,Linear))-Int32(1)
 if extra;CUDA.@cuassert gid<=Int32(prod(@ndrange()));end
 s=Int32(0)
 for n=Int32(1):gid
  s+=Int32(1);CUDA.@cuassert s<=gid
 end
end
backend=CUDABackend()
assertion_study(N->test_kernel(backend,32)(N;ndrange=64), k->perf_kernel(backend,256)(k==1;ndrange=256000))
