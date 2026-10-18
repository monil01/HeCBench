using CUDA
include("../study_common.jl")
function update_kernel(pos,vel,best,globalbest,P,rp,rg)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=P*Int32(30);move_particle_element(i,pos,vel,best,globalbest,rp,rg);end
 return
end
function best_kernel(pos,best,globalbest,P)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=P
  a,improved=candidate(i,pos,best)
  if improved
   base=(i-Int32(1))*Int32(30);@inbounds for j in 1:30;best[base+j]=a[j];end
   if source_fitness(a)<130f0
    for j in 1:30;CUDA.@atomic globalbest[j]+=a[j];end
   end
  end
 end
 return
end
function factory(pos,vel,best,globalbest,P,R)
 ()->begin
  for iter in Int32(0):R-Int32(1)
   rp=source_random_clamped(iter);rg=source_random_clamped(R-iter)
   @cuda threads=256 blocks=4 update_kernel(pos,vel,best,globalbest,P,rp,rg)
   @cuda threads=256 blocks=1 best_kernel(pos,best,globalbest,P)
  end
 end
end
main()=pso_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
