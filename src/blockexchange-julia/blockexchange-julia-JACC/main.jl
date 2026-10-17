using CUDA,JACC
function exchange_kernel!(index,out,input,n,workspace)
 t=mod(index-1,256);tile=div(index-1,256)
 base=tile*1024
 if base<n
  shared=JACC.shared(workspace)
  for j in 0:3;shared[4t+j+1]=input[base+4t+j+1];end
  JACC.sync_workgroup()
  for j in 0:3;out[base+4t+j+1]=shared[256j+t+1];end
 end
end
function launch(out,input,n,workspace)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=4096,sync=false),cld(n,256)*256,exchange_kernel!,out,input,n,workspace)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 4096;repeat=length(args)>1 ? parse(Int,args[2]) : 10;n%1024==0||Base.error("fulltiles required")
 input=Int32.(0:n-1);expected=Float64[input[div(i,1024)*1024+mod(i,4)*256+div(mod(i,1024),4)+1] for i in 0:n-1]
 di=CuArray(input);out=CUDA.zeros(Int32,n);workspace=CUDA.zeros(Int32,1024);launch(out,di,n,workspace);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns();for _ in 1:repeat;launch(out,di,n,workspace);end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
  emit(trial,seconds,Float64.(Array(out)),expected,repeat;case_times=[seconds])
 end
end
main(ARGS)
