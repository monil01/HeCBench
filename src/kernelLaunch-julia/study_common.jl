using Printf,SHA
const REPEAT=1000
function run_study(launch,launch_echo)
 cases=[KLArgs(ntuple(i->UInt8(mod(37*i+11*f,256)),n)) for f in 0:1 for n in (16,256,4096)]
 nullout=reinterpret(Core.LLVMPtr{UInt8,1},UInt64(0))
 for args in cases
  n=sizeof(args);@assert n in (16,256,4096)
  @assert fieldoffset(typeof(args),1)==0
  out=CUDA.fill(UInt8(0xa5),n+32)
  # Payload diagnostic is outside every primary timing window.
  launch_echo(args,view(out,17:n+16));CUDA.device_synchronize()
  raw=Array(out);@assert raw[17:n+16]==collect(args.bytes)
  @assert all(==(0xa5),raw[1:16]) && all(==(0xa5),raw[n+17:end])
  one=CUDA.fill(UInt8(0xa5),33)
  ptr=reinterpret(Core.LLVMPtr{UInt8,1},pointer(one,17))
  GC.@preserve one begin
   launch(args,ptr);CUDA.device_synchronize()
  end
  @assert Array(one)==vcat(fill(UInt8(0xa5),16),[args.bytes[1]],fill(UInt8(0xa5),16))
  launch(args,nullout);CUDA.device_synchronize()
 end
 for trial in 1:5
  times=Float64[]
  for args in cases
   CUDA.device_synchronize()
   start=time_ns()
   for _ in 1:REPEAT;launch(args,nullout);end
   CUDA.device_synchronize()
   push!(times,(time_ns()-start)*1e-9/REPEAT)
  end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":0,\"source_empty_kernel\":false,\"source_no_observable_numerical_outputs\":true,\"no_output_reason\":\"original_null_destination\",\"execution_checked\":true,\"numerical_agreement_kind\":\"no_numerical_outputs_execution_only\",\"max_abs_error\":null,\"max_rel_error\":null,\"numeric_outputs\":[],\"timing_region\":\"sum_six_null_destination_launch_batches_plus_device_completion_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":1000,\"component_time_seconds\":[%s],\"auxiliary_diagnostic\":\"untimed_full_argument_echo_and_guarded_first_byte_store_pass\"}\n",trial,sum(times),join(times,","))
 end
end
