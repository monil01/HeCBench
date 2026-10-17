using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function count_kernel(index,bases,numbers,val,eff)
 j=index
 if j<=100000
  @inbounds n=numbers[j]
  for cnt in 1:3
   value=mr_test(bases,cnt,n,eff);JACC.@atomic val[1]+=value
  end
 end
end
function evidence_kernel(index,bases,numbers,codes,eff,width)
 j=index
 if j<=100000
  @inbounds n=numbers[j]
  for cnt in 1:3
   @inbounds codes[(width-1)*300000+(j-1)*3+cnt]=UInt32(mr_test(bases,cnt,n,eff))
  end
 end
end
function factory(bases,numbers,val,codes,eff)
 spec=JACC.launch_spec(;threads=256,blocks=391,shmem_size=0,sync=false)
 launch=width->JACC.parallel_for(spec,100096,count_kernel,bases,numbers[width],val,eff)
 evaluate=()->begin;for width in 1:4;JACC.parallel_for(spec,100096,evidence_kernel,bases,numbers[width],codes,eff,width);end;end
 launch,evaluate
end
main()=mr_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
