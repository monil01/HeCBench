using CUDA,Printf,SHA
const NODES=262144
const ACCESSES=UInt32(1000000)
function pointer_study(launch,diagnose,model)
 d=CuArray{UInt64}(undef,NODES);base=UInt64(pointer(d));h=UInt64[base+UInt64(8*mod(i+16,NODES)) for i in 0:NODES-1];copyto!(d,h)
 sm=Int(attribute(device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
 root=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","pointerchase-julia-"*model,"diagnostics");mkpath(root)
 # Compile both paths and preserve full original 80-target launch/completion warmup.
 for target in 0:sm-1;launch(d,base,ACCESSES,UInt32(target),sm);CUDA.synchronize();end
 out=CUDA.zeros(UInt64,4);counts=UInt32[0,1,17,1000000];diagnose(d,base,out,counts);CUDA.synchronize()
 for trial in 1:5
  before=Array(d);@assert before==h
  ts=Float64[]
  for target in 0:sm-1
   t=time_ns();launch(d,base,ACCESSES,UInt32(target),sm);CUDA.synchronize();push!(ts,(time_ns()-t)*1e-9)
  end
  got=Array(d);ok=got==h;diagnose(d,base,out,counts);CUDA.synchronize();cursors=Array(out)
  offsets=UInt64[8*mod(UInt64(a)*16,NODES) for a in counts]
  ok &= cursors==base .+ offsets
  path=joinpath(root,"trial$(trial)_raw_graph_cursor.bin");write(path,vcat(UInt64[base],got,cursors));sha=bytes2hex(sha256(read(path)))
  # The source primary has no numerical store. These diagnostics remain auxiliary.
  @printf("POINTER_DIAGNOSTIC {\"trial\":%d,\"path\":\"%s\",\"sha256\":\"%s\",\"bytes\":%d,\"pointer_base\":%d,\"cursor_offsets_bytes\":%s,\"graph_nodes_checked\":%d,\"timed\":false}\n",trial,path,sha,filesize(path),base,string(offsets),NODES)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":0,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_all_SMcount_target_block_launches_and_completions\",\"jit_excluded\":true,\"internal_repeats\":1,\"source_empty_kernel\":false,\"source_no_observable_numerical_outputs\":true,\"no_output_reason\":\"source_final_pointer_assertion_only\",\"source_final_pointer_assertion_checked\":true,\"pointer_graph_checked\":%s,\"device_completion_checked\":true,\"nonnull_final_pointer_checked\":true,\"pointer_storage_bits\":64,\"execution_checked\":true,\"numerical_agreement_claim\":false,\"numeric_outputs\":[],\"numerical_outputs\":[],\"path_times_seconds\":%s,\"target_blocks\":%d,\"accesses_per_selected_block\":1000000,\"buffer_bytes\":2097152,\"host_inclusive_ns_per_access\":%.17g}\n",trial,sum(ts),ok ? "true" : "false",got==h ? "true" : "false",string(ts),sm,sum(ts)*1e9/(Float64(ACCESSES)*sm))
 end
end
