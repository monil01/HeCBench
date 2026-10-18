using CUDA,Printf
function assertion_study(launch_test,launch_perf)
 if "--negative" in ARGS
  launch_test(Int32(64));CUDA.device_synchronize(blocking=true)
  code=0
  try
   launch_test(Int32(60));CUDA.device_synchronize(blocking=true)
  catch e
   showerror(stderr,e);println(stderr)
   if e isa CUDA.CuError;code=Int(e.code);else;rethrow();end
  end
  println("ASSERT_STATUS ",code);flush(stdout);flush(stderr)
  ccall(:_exit,Cvoid,(Cint,),code==710 ? 0 : 3)
 end
 launch_perf(1);CUDA.device_synchronize(blocking=true)
 launch_perf(2);CUDA.device_synchronize(blocking=true)
 for trial=1:5
  times=zeros(2)
  for k=1:2
   CUDA.device_synchronize(blocking=true);start=time_ns()
   launch_perf(k);CUDA.device_synchronize(blocking=true)
   times[k]=(time_ns()-start)*1e-9
  end
  @printf("PERF_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"component_time_seconds\":[%.17g,%.17g]}\n",trial,sum(times),times[1],times[2]);flush(stdout)
 end
end
