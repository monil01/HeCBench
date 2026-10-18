using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const TEMPLATE=CUDA.zeros(UInt64,512)
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=64,blocks=128,shmem_size=4096,sync=false)
function ntt_jacc!(i,y,x,template)
 t=Int32((i-1)%64);b=Int32((i-1)÷64);buffer=JACC.shared(template)
 phase1!(x,buffer,b,t);JACC.sync_workgroup();phase2!(y,buffer,b,t);nothing
end
run_study((y,x)->JACC.parallel_for(ntt_jacc!,SPEC,8192,y,x,TEMPLATE),"jacc")
