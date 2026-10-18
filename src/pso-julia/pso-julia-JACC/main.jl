using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function update_kernel(index,pos,vel,best,globalbest,P,rp,rg)
 i=Int32(index)
 if i<=P*Int32(30);move_particle_element(i,pos,vel,best,globalbest,rp,rg);end

end
function best_kernel(index,pos,best,globalbest,P)
 i=Int32(index)
 if i<=P
  a,improved=candidate(i,pos,best)
  if improved
   base=(i-Int32(1))*Int32(30);@inbounds for j in 1:30;best[base+j]=a[j];end
   if source_fitness(a)<130f0
    for j in 1:30;JACC.@atomic globalbest[j]+=a[j];end
   end
  end
 end

end
function factory(pos,vel,best,globalbest,P,R)
 mapSpec=JACC.launch_spec(;threads=256,blocks=4,shmem_size=0,sync=false);bestSpec=JACC.launch_spec(;threads=256,blocks=1,shmem_size=0,sync=false)
 ()->begin
  for iter in Int32(0):R-Int32(1)
   rp=source_random_clamped(iter);rg=source_random_clamped(R-iter)
   JACC.parallel_for(mapSpec,1024,update_kernel,pos,vel,best,globalbest,P,rp,rg)
   JACC.parallel_for(bestSpec,256,best_kernel,pos,best,globalbest,P)
  end
 end
end
main()=pso_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
