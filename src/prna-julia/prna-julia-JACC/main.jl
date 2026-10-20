using CUDA, JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
# This resource seed is not a mathematical/oracle input. Every active reduction
# overwrites all128sharedlocations with freshly computed GPU partial energies.
const PRNA_SCRATCH_SEED=CUDA.zeros(Float32,128)
const PRNA_INIT_SPEC=JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false)
const PRNA_SCALAR_SPECS=Dict(n=>JACC.launch_spec(threads=1,blocks=n,shmem_size=0,sync=false) for n in (292,7,9,27))
const PRNA_PARTIAL_SPECS=Dict(n=>JACC.launch_spec(threads=128,blocks=n,shmem_size=512,sync=false) for n in (292,7,9,27))
const PRNA_FINAL_SPEC=JACC.launch_spec(threads=128,blocks=1,shmem_size=512,sync=false)
function rna_init_callback(idx,c,n)
 c.w5[1]=c.w5[2]=c.w3[n]=c.w3[n+Int32(1)]=0f0
 nothing
end
function rna_scalar_callback(idx,stage,c,n,d)
 rna_scalar_stage(stage,c,n,d,Int32(idx)-Int32(1))
 nothing
end
@inline function rna_jacc_reduce(value,tid,scratch)
 @inbounds scratch[tid+Int32(1)]=value
 JACC.sync_workgroup();stride=Int32(64)
 while stride>0
  if tid<stride
   @inbounds scratch[tid+Int32(1)]=free_energy_sum(scratch[tid+Int32(1)],scratch[tid+stride+Int32(1)])
  end
  JACC.sync_workgroup();stride=div(stride,Int32(2))
 end
 result=tid==0 ? (@inbounds scratch[1]) : value
 JACC.sync_workgroup()
 result
end
function rna_partial_callback(idx,stage::Val{S},c,n,d,seed) where S
 linear=Int32(idx)-Int32(1);i=div(linear,Int32(128));lane=rem(linear,Int32(128))
 tx=S==2 ? rem(lane,Int32(8)) : lane
 ty=S==2 ? div(lane,Int32(8)) : Int32(0)
 tid=S==2 ? tx*Int32(16)+ty : lane
 a,b,action=rna_partial(stage,c,n,d,i,tx,ty,lane)
 if action==2
  scratch=JACC.shared(seed)
  a=rna_jacc_reduce(a,tid,scratch)
  if S==6||S==8;b=rna_jacc_reduce(b,tid,scratch);end
 end
 if tid==0;rna_finish(stage,c,n,d,i,a,b,action);end
 nothing
end
launch_init(c,n)=JACC.parallel_for(PRNA_INIT_SPEC,1,rna_init_callback,c,n)
function launch_stage!(s::Val{S},c,n,d) where S
 if S==1||S==4||S==7
  JACC.parallel_for(PRNA_SCALAR_SPECS[Int(n)],Int(n),rna_scalar_callback,s,c,n,d)
 else
  blocks=S==8 ? 1 : Int(n);spec=S==8 ? PRNA_FINAL_SPEC : PRNA_PARTIAL_SPECS[Int(n)]
  JACC.parallel_for(spec,128*blocks,rna_partial_callback,s,c,n,d,PRNA_SCRATCH_SEED)
 end
 nothing
end
function report_resource_seed(phase)
 CUDA.device_synchronize(blocking=true);a=Array(PRNA_SCRATCH_SEED);@assert all(iszero,a)
 print("STUDY_RESOURCE_INPUT {\"name\":\"JACC_shared_scratch_seed_not_mathematical_input\",\"phase\":\"",phase,"\",\"bytes\":512,\"uint32_words\":[")
 for(k,x)in enumerate(reinterpret(UInt32,a));print(k==1 ? "" : ",",x);end
 println("],\"all_zero\":true,\"timed\":false}");flush(stdout)
end
report_resource_seed("before_all_trials")
prna_study()
report_resource_seed("after_all_trials")
