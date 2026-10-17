const BT=128;const CC=64;const OC=192
function study(run!)
 a=Float32[Float32(mod(i,17)-8)/16f0 for i in 0:BT*CC-1];w=Float32[Float32(mod(i,13)-6)/16f0 for i in 0:OC*CC-1];b=Float32[Float32(mod(i,7)-3)/8f0 for i in 0:OC-1]
 # Dyadic inputs imply exact Float32 accumulation; independent Int64 numerator oracle.
 want=Float32[Float32(32*(mod(o,7)-3)+sum((mod(t*CC+k,17)-8)*(mod(o*CC+k,13)-6) for k in 0:CC-1))/256f0 for t in 0:BT-1 for o in 0:OC-1]
 A=CuArray(a);W=CuArray(w);B=CuArray(b);O=CUDA.zeros(Float32,BT*OC)
 run!(O,A,W,B);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();t=time_ns();for k in 1:100;run!(O,A,W,B);CUDA.synchronize();end;elapsed=(time_ns()-t)*1e-9/100
  got=Array(O);err=maximum(abs.(got.-want));ok=all(isfinite,got)&&err<=.1;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"default1_naive_matmul_bias_host_sync_mean\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("QKV integer-dot oracle failed")
 end
end
