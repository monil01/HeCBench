using CUDA,KernelAbstractions,Atomix
include("../study_common.jl")
@kernel function update_kernel(pos,vel,best,globalbest,P,rp,rg)
 i=Int32(@index(Global,Linear))
 if i<=P*Int32(30);move_particle_element(i,pos,vel,best,globalbest,rp,rg);end

end
@kernel function best_kernel(pos,best,globalbest,P)
 i=Int32(@index(Global,Linear))
 if i<=P
  a,improved=candidate(i,pos,best)
  if improved
   base=(i-Int32(1))*Int32(30);@inbounds for j in 1:30;best[base+j]=a[j];end
   if source_fitness(a)<130f0
    for j in 1:30;Atomix.@atomic globalbest[j]+=a[j];end
   end
  end
 end

end
function factory(pos,vel,best,globalbest,P,R)
 update=update_kernel(CUDABackend(),256);bestlaunch=best_kernel(CUDABackend(),256)
 ()->begin
  for iter in Int32(0):R-Int32(1)
   rp=source_random_clamped(iter);rg=source_random_clamped(R-iter)
   update(pos,vel,best,globalbest,P,rp,rg;ndrange=1024)
   bestlaunch(pos,best,globalbest,P;ndrange=256)
  end
 end
end
main()=pso_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
