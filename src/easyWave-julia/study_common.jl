using CUDA,Printf
const EWROOT=@__DIR__
const EWNX=Int32(23);const EWNY=Int32(19);const EWSTEPS=8;const EWN=23*19*12
function easywave_study(factory)
 prepared=map((1,2)) do case
  fixture=collect(reinterpret(Float32,read(joinpath(EWROOT,"study_case$(case)_f32.bin"))))
  oracle=collect(reinterpret(Float64,read(joinpath(EWROOT,"oracle_case$(case)_f64.bin"))))
  sizes=(EWN,20,24,20,24,20);bases=map(n->CUDA.fill(-8192f0,n+32),sizes);a=map((b,n)->view(b,17:16+n),bases,sizes)
  ibases=ntuple(_->CUDA.fill(Int32(-8192),33),4);B=map(x->view(x,17:17),ibases);hB=ntuple(_->zeros(Int32,1),4)
  cursor=1;for k in eachindex(a);copyto!(a[k],fixture[cursor:cursor+sizes[k]-1]);cursor+=sizes[k];end
  node,R6=a[1:2];C=a[3:6];launch=factory(node,R6,C,B,EWNY,EWNX);initial=case==1 ? (Int32(5),Int32(5),Int32(19),Int32(15)) : (Int32(2),Int32(2),Int32(22),Int32(18))
  function pipeline(capture)
   b=initial;states=Float64[];bt=Float64[]
   for step in 0:EWSTEPS-1
    for stage in 1:4
     if stage==4
      for k in 1:4;hB[k][1]=b[k];copyto!(B[k],hB[k]);end
     end
     launch(stage,b,Int32(step))
     if stage==4
      for k in 1:4;copyto!(hB[k],B[k]);end
      b=ntuple(k->hB[k][1],4);capture&&append!(bt,b)
     end
     capture&&append!(states,Float64.(Array(node)))
    end
   end
   CUDA.device_synchronize(blocking=true)
   states,bt,b
  end
  function guards()
   raw=Float64[]
   for base in (bases...,ibases...);v=Array(base);append!(raw,v[1:16]);append!(raw,v[end-15:end]);end
   raw
  end
  reset=()->copyto!(node,fixture[1:EWN])
  reset();pipeline(false)
  (;case,fixture,oracle,a,bases,ibases,B,initial,pipeline,guards,reset)
 end
 CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  prefix=Float64[];outputs=Float64[];guardvalues=Float64[];times=Float64[];ok=true;ma=0.;mr=0.;normalized=0.
  for z in prepared
   z.reset();CUDA.device_synchronize(blocking=true)
   append!(prefix,Float64.(Array(z.a[1])));for x in z.a[2:6];append!(prefix,Float64.(Array(x)));end
   append!(prefix,Float64[EWNX,EWNY,EWSTEPS,z.initial...,1f-6,1f-4,1f-4,1,16,16])
   start=time_ns();_,_,timed_bounds=z.pipeline(false);push!(times,(time_ns()-start)*1e-9)
   timed_final=Float64.(Array(z.a[1]));append!(guardvalues,z.guards())
   z.reset();states,bt,captured_bounds=z.pipeline(true);actual=vcat(timed_final,states,bt);append!(outputs,actual);append!(guardvalues,z.guards())
   e=abs.(actual.-z.oracle);ma=max(ma,maximum(e));mr=max(mr,maximum(e./max.(abs.(z.oracle),eps(Float64))));normalized=max(normalized,maximum(e./(1e-7.+1e-5.*abs.(z.oracle))));ok&=all(isfinite,actual)&&normalized<=1&&timed_bounds==captured_bounds&&actual[end-31:end]==z.oracle[end-31:end]
   if z.case==2;ok&=timed_final==Float64.(z.fixture[1:EWN]);end
  end
  ok&=all(==(-8192),guardvalues);allok&=ok;artifact=vcat(prefix,outputs,guardvalues);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":347448,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"relative_error_floor\":2.220446049250313e-16,\"max_scaled_oracle_error\":%.12g,\"timing_region\":\"sum_two_8step_shallow_water_four_kernel_pipelines_with_adaptive_bounds_H2D_D2H\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_steps\":8,\"timed_cases\":2,\"untimed_full_stage_capture_reexecution\":true,\"component_time_seconds\":[%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",ma,mr,normalized,times[1],times[2],path,length(artifact))
 end
 allok||error("Full shallow-water pipeline/independent oracle/guards mismatch");println("PASS")
end
