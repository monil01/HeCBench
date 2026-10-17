using CUDA,KernelAbstractions,Atomix
include("../study_common.jl")
@kernel function count_kernel(bases,numbers,val,eff)
 j=@index(Global,Linear)
 if j<=100000
  @inbounds n=numbers[j]
  for cnt in 1:3
   value=mr_test(bases,cnt,n,eff);Atomix.@atomic val[1]+=value
  end
 end
 return
end
@kernel function evidence_kernel(bases,numbers,codes,eff,width)
 j=@index(Global,Linear)
 if j<=100000
  @inbounds n=numbers[j]
  for cnt in 1:3
   @inbounds codes[(width-1)*300000+(j-1)*3+cnt]=UInt32(mr_test(bases,cnt,n,eff))
  end
 end
 return
end
function factory(bases,numbers,val,codes,eff)
 count=count_kernel(CUDABackend(),256);evidence=evidence_kernel(CUDABackend(),256)
 launch=width->count(bases,numbers[width],val,eff;ndrange=100096)
 evaluate=()->begin;for width in 1:4;evidence(bases,numbers[width],codes,eff,width;ndrange=100096);end;end
 launch,evaluate
end
main()=mr_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
