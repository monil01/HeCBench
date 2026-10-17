using CUDA
include("../study_common.jl")
function count_kernel(bases,numbers,val,eff)
 j=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if j<=100000
  @inbounds n=numbers[j]
  for cnt in 1:3
   value=mr_test(bases,cnt,n,eff);CUDA.@atomic val[1]+=value
  end
 end
 return
end
function evidence_kernel(bases,numbers,codes,eff,width)
 j=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if j<=100000
  @inbounds n=numbers[j]
  for cnt in 1:3
   @inbounds codes[(width-1)*300000+(j-1)*3+cnt]=UInt32(mr_test(bases,cnt,n,eff))
  end
 end
 return
end
function factory(bases,numbers,val,codes,eff)
 launch=width->begin;@cuda threads=256 blocks=391 count_kernel(bases,numbers[width],val,eff);end
 evaluate=()->begin;for width in 1:4;@cuda threads=256 blocks=391 evidence_kernel(bases,numbers[width],codes,eff,width);end;end
 launch,evaluate
end
main()=mr_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
