function loadf(name)
 open(joinpath(@__DIR__,name)) do io
  read!(io,Vector{Float64}(undef,filesize(io)÷8))
 end
end
function study(allocate,launch)
 f=loadf("fixtures_f64.bin");oracle=loadf("oracle_f64.bin");budget=loadf("error_budget_f64.bin");pos=2;cases=Any[];sizes=(3936,3936,3936,288,32,8)
 for _ in 1:Int(f[1])
  repeats=Int(f[pos]);zero=Int(f[pos+1]);pos+=2;inputs=Vector{Vector{Float64}}(undef,6)
  for k in (1,3,4,5,6)
   inputs[k]=f[pos:pos+sizes[k]-1];pos+=sizes[k]
  end
  inputs[2]=copy(inputs[1]);arrays=[allocate(vcat(-12345.,v,-12345.)) for v in inputs];template=allocate(zeros(Float64,56));push!(cases,(repeats=repeats,zero=zero,inputs=inputs,arrays=arrays,template=template))
 end
 function reset(c)
  for k in 1:2
   copyto!(c.arrays[k],vcat(-12345.,c.inputs[k],-12345.))
  end
  CUDA.device_synchronize(blocking=true)
 end
 function pipeline(c)
  src=c.arrays[1];dst=c.arrays[2]
  for _ in 1:c.repeats,i in 1:11,j in 1:40
   launch(i,j,c,src,dst);src,dst=dst,src
  end
  CUDA.device_synchronize(blocking=true)
 end
 for c in cases
  reset(c);pipeline(c)
 end
 for trial in 1:5
  payload=copy(f);times=Float64[];total=0.;ok=true
  for c in cases
   reset(c);start=time_ns();pipeline(c);tm=(time_ns()-start)*1e-9;total+=tm;push!(times,tm)
   for a in c.arrays
    append!(payload,Array(a))
   end
   ok&=all(iszero,Array(c.template))
  end
  error=abs.(payload[length(f)+1:end].-oracle);ok&=all(isfinite,payload)&&all(error.<=budget);maxratio=maximum(e>0 ? (b>0 ? e/b : 1e300) : 0. for (e,b) in zip(error,budget));path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",maximum(error),",\"max_rel_error\":null,\"max_error_budget_ratio\":",maxratio,",\"timing_region\":\"sum3_defined_storage_source_440cell_swap_pipelines_context_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":1760,\"case_times_seconds\":[",join(times,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("Fullsourceoperator/source-roundingbudget/readonly/guards failed")
 end
end
