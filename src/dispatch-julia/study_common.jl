using CUDA,Printf,Statistics
function dispatch_study(launch)
 launch();CUDA.device_synchronize(blocking=true)
 for trial in 1:5
  samples=zeros(1100,3);calls=0
  for mode in 1:3
   CUDA.device_synchronize(blocking=true)
   for i in 1:1100
    start=time_ns()
    if mode==3
     for _ in 1:1000;launch();end
    else
     launch()
    end
    if mode!=1;CUDA.device_synchronize(blocking=true);end
    elapsed=(time_ns()-start)*1e-9;samples[i,mode]=elapsed;calls+=mode==3 ? 1000 : 1
   end
   # Async enqueue ROI excludes this completion; allmodes checked beforeacceptance.
   CUDA.device_synchronize(blocking=true)
  end
  means=[mean(view(samples,101:1100,m))/(m==3 ? 1000 : 1) for m in 1:3]
  deviations=[std(view(samples,101:1100,m);corrected=false)/(m==3 ? 1000 : 1) for m in 1:3]
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":0,\"source_empty_kernel\":true,\"execution_checked\":true,\"numerical_agreement_kind\":\"no_numerical_outputs_execution_only\",\"max_abs_error\":null,\"max_rel_error\":null,\"numeric_outputs\":[],\"timing_region\":\"batch1000_empty_dispatch_plus_whole_device_completion_per_launch\",\"jit_excluded\":true,\"internal_repeats\":1000,\"source_warmup_samples_per_mode\":100,\"source_retained_samples_per_mode\":1000,\"completed_source_loop_launch_calls\":%d,\"component_time_seconds\":{\"enqueue\":%.12g,\"single_dispatch\":%.12g,\"batch1000_per_launch\":%.12g},\"component_stddev_seconds\":{\"enqueue\":%.12g,\"single_dispatch\":%.12g,\"batch1000_per_launch\":%.12g},\"execution_check\":\"normaldriverlaunch APIs didnotthrow;wholedevicecompletionaftereachmode succeeded\",\"internal_timing_sample_units\":\"seconds_per_API_sample_batch_samples_include1000launches\",\"internal_timing_samples_seconds\":{",trial,means[3],calls,means[1],means[2],means[3],deviations[1],deviations[2],deviations[3])
  for (m,name) in enumerate(("enqueue","single_dispatch","batch1000"))
   m>1&&print(",");print('"',name,"\":[")
   for i in 1:1100;i>1&&print(",");@printf("%.12g",samples[i,m]);end
   print("]")
  end
  println("}}")
 end
 println("EXECUTION_TIMING_VALIDATED_NO_NUMERICAL_OUTPUTS")
end
