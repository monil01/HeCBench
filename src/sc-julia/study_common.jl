using Libdl
const SENT=Int32(-777)
const hostlib=Libdl.dlopen(joinpath(pwd(),"libsc_cpu.so"))
const createfn=Libdl.dlsym(hostlib,:sc_host_create)
const datafn=Libdl.dlsym(hostlib,:sc_host_data)
const flagsfn=Libdl.dlsym(hostlib,:sc_host_flags)
const resetfn=Libdl.dlsym(hostlib,:sc_host_reset)
const startfn=Libdl.dlsym(hostlib,:sc_host_start)
const waitfn=Libdl.dlsym(hostlib,:sc_host_wait)
const destroyfn=Libdl.dlsym(hostlib,:sc_host_destroy)
function loadints(name)
 p=joinpath(@__DIR__,name);filesize(p)%4==0||error("file size")
 open(io->read!(io,Vector{Int32}(undef,filesize(p)÷4)),p)
end
function study(allocate,launch)
 f=loadints("fixtures_i32.bin");oracle=loadints("oracle_i32.bin");pos=2;cases=Any[]
 for ci in 1:Int(f[1])
  n,ai=Int.(f[pos:pos+1]);pos+=2;input=copy(f[pos:pos+n-1]);prefixpos=pos;pos+=n;tasks=cld(n,8192);cut=trunc(Int,tasks*(Float32(ai)*.1f0));padded=tasks*8192;gpad=(tasks-cut)*8192
  host=ccall(createfn,Ptr{Cvoid},(Cint,Cint),padded,tasks);host==C_NULL&&error("host allocation")
  hp=ccall(datafn,Ptr{Int32},(Ptr{Cvoid},),host);fp=ccall(flagsfn,Ptr{Int32},(Ptr{Cvoid},),host)
  data=unsafe_wrap(Array,hp,(padded,);own=false);flags=unsafe_wrap(Array,fp,(tasks+1,);own=false)
  db=allocate(fill(SENT,gpad+2));fb=allocate(fill(SENT,tasks+3));template=allocate(zeros(Int32,257))
  push!(cases,(n=n,ai=ai,input=input,prefixpos=prefixpos,tasks=tasks,cut=cut,padded=padded,gpad=gpad,host=host,hp=hp,fp=fp,data=data,flags=flags,db=db,fb=fb,d=view(db,2:gpad+1),gf=view(fb,2:tasks+2),template=template))
 end
 pos==length(f)+1||error("trailing fixture")
 function reset(c)
  GC.@preserve c ccall(resetfn,Cvoid,(Ptr{Cvoid},Ptr{Int32},Cint,Cint),c.host,c.input,c.n,c.cut)
  copyto!(c.db,vcat(Int32[SENT],c.data[c.cut*8192+1:c.padded],Int32[SENT]));copyto!(c.fb,vcat(Int32[SENT],c.flags,Int32[SENT]));CUDA.device_synchronize(blocking=true)
 end
 function run(c)
  launch(c.d,c.gf,c.n,c.tasks,c.cut,c.template)
  ccall(startfn,Cvoid,(Ptr{Cvoid},Cint,Cfloat),c.host,c.n,Float32(c.ai)*.1f0)
  CUDA.device_synchronize(blocking=true);ccall(waitfn,Cvoid,(Ptr{Cvoid},),c.host)
 end
 for c in cases;reset(c);run(c);end
 for trial in 1:5
  payload=copy(f);results=Int32[];total=0.;times=Float64[];ok=true
  for c in cases
   reset(c);before=Array(c.d);payload[c.prefixpos:c.prefixpos+c.n-1]=c.data[1:c.n];logicalgpu=c.n-c.cut*8192
   if logicalgpu>0;payload[c.prefixpos+c.cut*8192:c.prefixpos+c.n-1]=before[1:logicalgpu];end
   if logicalgpu<c.gpad;ok &=all(iszero,before[max(logicalgpu,0)+1:end]);end
   start=time_ns();run(c);elapsed=(time_ns()-start)*1e-9;total+=elapsed;push!(times,elapsed)
   gf=Array(c.gf);cpu=c.cut==0 ? 0 : Int(c.flags[c.cut+1])-2;gpu=Int(gf[end])-1;count=cpu+gpu;0<=count<=c.n||error("bad count")
   c.gpad>0&&(c.data[cpu+1:cpu+c.gpad]=Array(c.d))
   push!(results,Int32(count));append!(results,c.data[1:count]);append!(results,c.flags);append!(results,gf)
   append!(results,Int32[unsafe_load(c.hp,0),unsafe_load(c.hp,c.padded+1),unsafe_load(c.fp,0),unsafe_load(c.fp,c.tasks+2)])
   append!(results,Array(c.db)[[1,end]]);append!(results,Array(c.fb)[[1,end]])
   ok &=all(iszero,Array(c.template))
  end
  ok &=payload==f&&results==oracle;append!(payload,results);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum15_static_CPU_GPU_compaction_compute_threadspawn_contextjoin\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":15,\"CPU_workers_per_case\":4,\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("Fullcompaction/count/flags/guards oracle mismatch")
 end
 for c in cases;ccall(destroyfn,Cvoid,(Ptr{Cvoid},),c.host);end
end
