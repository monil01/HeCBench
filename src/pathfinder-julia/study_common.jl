function loadints(name)
 open(joinpath(@__DIR__,name)) do io
  read!(io,Vector{Int32}(undef,filesize(io)÷4))
 end
end
function study(allocate,launch)
 f=loadints("fixtures_i32.bin");oracle=loadints("oracle_i32.bin");pos=2;cases=Any[]
 for _ in 1:f[1]
  cols,rows,h=Int.(f[pos:pos+2]);pos+=3;data=f[pos:pos+cols*rows-1];pos+=cols*rows
  wall=allocate(vcat(Int32[-12345],data[cols+1:end],Int32[-12345]));a=allocate(fill(Int32(-12345),cols+2));b=allocate(fill(Int32(-12345),cols+2));dbg=allocate(zeros(Int32,16386));template=allocate(zeros(Int32,500))
  push!(cases,(cols=cols,rows=rows,h=h,data=data,wall=wall,a=a,b=b,dbg=dbg,template=template))
 end
 function reset(c)
  host=vcat(Int32[-12345],c.data[1:c.cols],Int32[-12345]);copyto!(c.a,host);copyto!(c.b,host);fill!(c.dbg,Int32(0));c.dbg[1:1].=Int32(-12345);c.dbg[end:end].=Int32(-12345);CUDA.device_synchronize(blocking=true)
 end
 function pipeline(c)
  src=c.a;dst=c.b
  for t in 0:c.h:c.rows-2
   it=min(c.h,c.rows-1-t);launch(c.wall,src,dst,c.dbg,it,c.h,c,t,c.template);src,dst=dst,src
  end
  CUDA.device_synchronize(blocking=true);(src,dst)
 end
 for c in cases
  reset(c);pipeline(c)
 end
 for trial in 1:5
  payload=copy(f);total=0.;times=Float64[];ok=true
  for c in cases
   reset(c);start=time_ns();src,dst=pipeline(c);elapsed=(time_ns()-start)*1e-9;total+=elapsed;push!(times,elapsed)
   for array in (c.wall,src,dst,c.dbg)
    append!(payload,Array(array))
   end
   ok &=all(iszero,Array(c.template))
  end
  ok &=payload[length(f)+1:end]==oracle;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum5_source_ghostzone_pipelines_context_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":60,\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullDP/debug/readonly/guards differ from independentoracle")
 end
end
