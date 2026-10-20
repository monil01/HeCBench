const GUARD=-8191.25
readf(p)=reinterpret(Float64,read(p)) |> collect
function fields(a,b,n)
 atol=(2e-6,5e-3,5e-9,5e-9,5e-9,2e-7,2e-7,2e-7,2e-7,2e-7,2e-7,5e-4,5e-4,5e-4)
 for i=0:n-1
  a[14i+1]>0 || return false
  for j=1:14
   x=a[14i+j];y=b[14i+j]
   isfinite(x) && abs(x-y)<=atol[j]+1e-9*abs(y) || return false
  end
 end
 true
end
samebits(a,b)=reinterpret(UInt64,a)==reinterpret(UInt64,b)
function emit_trial(trial,seconds,ok,prefix,out,s,ns,timing)
 path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");open(path,"w") do io;write(io,prefix);write(io,out);end
 tm=timing ? string(seconds) : "null"
 println("STUDY_TRIAL {\"trial\":$trial,\"time_seconds\":$tm,\"correct\":$ok,\"checked_values\":$(length(out)),\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"four_source_SPH_kernels_all_sequential_steps_and_wholedevicecompletion_perstep_average\",\"timing_valid\":$timing,\"jit_excluded\":true,\"input_prefix_count\":$(length(prefix)),\"source_step_count\":$(s.steps),\"internal_repeats\":1,\"timed_kernel_launches\":$(4*s.steps),\"step_semantics\":\"sequential_coupled_trajectory_not_independent_repetitions\",\"untimed_warmup_trajectory_executions_per_process\":1,\"warmup_checkpoints\":$ns,\"same_single_warmup_trace_in_each_trial_artifact\":true,\"all_timed_intermediate_states_archived\":false,\"output_artifact\":{\"path\":\"$path\",\"dtype\":\"float64\",\"count\":$(length(prefix)+length(out))}}")
 flush(stdout)
end
function study_main()
 name=only(ARGS);f=readf(name*"_initial_f64.bin");b=readf(name*"_boundary_f64.bin");rp=readf(name*"_params_raw.bin");s=only(reinterpret(SPHParams,rp));nf=length(f);nb=length(b);np=length(rp);cpu=readf(name*"_CPU_final_f64.bin");trace=readf(name*"_CPU_checkpoints_f64.bin");steps=name=="default" ? [1,2,10,50,100,250,500] : [1,2,17]
 append!(f,fill(GUARD,17));append!(b,fill(GUARD,17));append!(rp,fill(GUARD,17));CUDA.stream!(CUDA.legacy_stream());df=CuArray(f);db=CuArray(b);raw=CuArray(rp);dp=unsafe_wrap(CuArray,CuPtr{SPHParams}(UInt(pointer(raw))),(1,);own=false)
 reset!()=(copyto!(df,f);copyto!(db,b);copyto!(raw,rp);CUDA.device_synchronize(blocking=true);nothing)
 function inputs()
  x=Array(df);y=Array(db);z=Array(raw);@assert samebits(x,f)&&samebits(y,b)&&samebits(z,rp);vcat(x,y,z)
 end
 reset!();wp=inputs();warm=Float64[];at=1;warmok=true
 for step=1:Int(s.steps)
  launch!(df,db,dp,s)
  if at<=length(steps)&&step==steps[at]
   CUDA.device_synchronize(blocking=true);x=Array(df);warmok &= fields(x,view(trace,(at-1)*nf+1:at*nf),Int(s.nf))&&all(==(GUARD),view(x,nf+1:length(x)));append!(warm,x);at+=1
  end
 end
 CUDA.device_synchronize(blocking=true);@assert at==length(steps)+1
 if !warmok;emit_trial(0,nothing,false,wp,warm,s,length(steps),false);exit(1);end
 for trial=1:5
  reset!();prefix=inputs();CUDA.device_synchronize(blocking=true);t=time_ns()
  for step=1:Int(s.steps);launch!(df,db,dp,s);end
  CUDA.device_synchronize(blocking=true);seconds=(time_ns()-t)*1e-9/Int(s.steps);x=Array(df);y=Array(db);z=Array(raw)
  ok=fields(x,cpu,Int(s.nf))&&all(==(GUARD),view(x,nf+1:length(x)))&&samebits(y,b)&&samebits(z,rp);emit_trial(trial,seconds,ok,prefix,vcat(x,y,z,warm),s,length(steps),true);ok||exit(1)
 end
 CUDA.unsafe_free!(df);CUDA.unsafe_free!(db);CUDA.unsafe_free!(raw)
end
