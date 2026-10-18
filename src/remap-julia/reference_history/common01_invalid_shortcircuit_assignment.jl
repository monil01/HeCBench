using Libdl
const G=64
const SENT=Int32(-123456789)
const vendor=Libdl.dlopen(joinpath(pwd(),"libremap_vendor.so"))
const sequence_fn=Libdl.dlsym(vendor,:remap_sequence)
const sort_fn=Libdl.dlsym(vendor,:remap_sort)
const unique_fn=Libdl.dlsym(vendor,:remap_unique)
const version_fn=Libdl.dlsym(vendor,:remap_thrust_version)
const runtime_fn=Libdl.dlsym(vendor,:remap_runtime_version)
function loadints(name)
 path=joinpath(@__DIR__,name);filesize(path)%4==0||error("malformed file")
 open(path) do io
  read!(io,Vector{Int32}(undef,filesize(path)÷4))
 end
end
ptr(a)=Ptr{Int32}(UInt(pointer(a)))
function checkvendor(status)
 status==0||error("Thrust shim status $status")
end
function study(allocate,launch)
 f=loadints("fixtures_i32.bin");oracle=loadints("oracle_i32.bin");pos=2;cases=Any[]
 for ci in 1:Int(f[1])
  n=Int(f[pos]);pos+=1;input=copy(f[pos:pos+n-1]);bases=ntuple(_->allocate(fill(SENT,n+2G)),4);arrays=map(b->view(b,G+1:G+n),bases)
  push!(cases,(n=n,input=input,prefix_pos=pos,bases=bases,arrays=arrays));pos+=n
 end
 pos==length(f)+1||error("fixture trailing bytes")
 function reset(c)
  for b in c.bases;fill!(b,SENT);end
  copyto!(c.arrays[1],c.input);CUDA.device_synchronize(blocking=true)
 end
 function cycle(c,times)
  keys,first,second,out=c.arrays
  stream=Ptr{Cvoid}(UInt(Base.unsafe_convert(CUDA.CUstream,CUDA.stream())))
  function timed(stage,fn)
   CUDA.device_synchronize(blocking=true);start=time_ns();fn();CUDA.device_synchronize(blocking=true);times[stage]+=(time_ns()-start)*1e-9
  end
  timed(1,()->checkvendor(ccall(sequence_fn,Cint,(Ptr{Int32},Ptr{Int32},Cint,Ptr{Cvoid}),ptr(first),ptr(second),c.n,stream)))
  timed(2,()->checkvendor(ccall(sort_fn,Cint,(Ptr{Int32},Ptr{Int32},Cint,Ptr{Cvoid}),ptr(keys),ptr(first),c.n,stream)))
  sorted=Array(keys);perm=Array(first)
  ok=sort(perm)==collect(Int32,0:c.n-1)
  ok&&=(c.input[Int.(perm).+1]==sorted)
  count=Ref{Cint}(0)
  timed(3,()->checkvendor(ccall(unique_fn,Cint,(Ptr{Int32},Ptr{Int32},Cint,Ref{Cint},Ptr{Cvoid}),ptr(keys),ptr(second),c.n,count,stream)))
  k=Int(count[]);1<=k<=c.n||error("invalid unique count")
  timed(4,()->launch(second,first,out,c.n,k))
  result=vcat(sorted,Int32[k],Array(view(keys,1:k)),Array(view(second,1:k)),Array(out))
  ok&=Array(first)==perm
  for b in c.bases;append!(result,Array(view(b,1:G)));append!(result,Array(view(b,length(b)-G+1:length(b))));end
  result,perm,ok
 end
 println("VENDOR thrust=",ccall(version_fn,Cint,())," runtime=",ccall(runtime_fn,Cint,())," library=",Libdl.dlpath(vendor)," stream=CUDA.current_task_stream completion=context")
 println("LOADED_LIBRARIES ",join(filter(p->occursin("cudart",p),Libdl.dllist()),";"))
 for c in cases;reset(c);cycle(c,zeros(4));end
 for trial in 1:5
  payload=copy(f);results=Int32[];times=zeros(4);perms=Vector{Vector{Int32}}();ok=true
  for c in cases
   reset(c);payload[c.prefix_pos:c.prefix_pos+c.n-1]=Array(c.arrays[1]);r,p,correct=cycle(c,times);append!(results,r);push!(perms,p);ok&=correct
  end
  ok &=payload==f && results==oracle;append!(payload,results)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum8_context_completed_sequence_sort_unique_authored_remap_stages\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_authored_kernel_calls\":8,\"stage_times_seconds\":[",join(times,","),"],\"permutation_witness\":[",join(["["*join(p,",")*"]" for p in perms],","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("Full rank/sorted/unique/starts/guard/permutation oracle check failed")
 end
end
