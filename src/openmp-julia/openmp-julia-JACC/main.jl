using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,a,b,repeat)
 for j in Int32(0):repeat-Int32(1)
  @inbounds a[i]+=mod(j,b)
 end
end
run!(a,b,r)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=length(a)÷256,shmem_size=0,sync=false),length(a),kernel!,a,b,r)
study(run!)
