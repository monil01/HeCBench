using CUDA,JACC
function prefix_kernel!(index,output,input,sums,::Val{WINDOW},::Val{WRITE_SUM},workspace) where {WINDOW,WRITE_SUM}
 t=mod(index-1,div(WINDOW,2));group=div(index-1,div(WINDOW,2))
 shared=JACC.shared(workspace)
 base=group*WINDOW
 shared[2t+1]=input[base+2t+1];shared[2t+2]=input[base+2t+2]
 JACC.sync_workgroup()
 cache0=shared[1];cache1=cache0+shared[2]
 stride=1
 # Preserve native block_size=256, including extra idle stages for16-element top prefix.
 while stride<256
  if 2t>=stride
   cache0=shared[2t-stride+1]+shared[2t+1];cache1=shared[2t+1-stride+1]+shared[2t+2]
  end
  JACC.sync_workgroup()
  shared[2t+1]=cache0;shared[2t+2]=cache1
  JACC.sync_workgroup()
  stride*=2
 end
 if WRITE_SUM&&t==0;sums[group+1]=shared[WINDOW];end
 output[base+2t+1]=t==0 ? 0f0 : shared[2t]
 output[base+2t+2]=shared[2t+1]
end
function addition_kernel!(index,output,prefix,workspace)
 tid=mod(index-1,256)+1;group=div(index-1,256)+1
 value=JACC.shared(workspace)
 if tid==1;value[1]=prefix[group];end
 JACC.sync_workgroup()
 output[index]+=value[1]
end
function pipeline!(output,input,sums,top,workspaces)
 JACC.parallel_for(JACC.launch_spec(threads=128,blocks=16,shmem_size=1024,sync=false),2048,prefix_kernel!,output,input,sums,Val(256),Val(true),workspaces[1])
 JACC.parallel_for(JACC.launch_spec(threads=8,blocks=1,shmem_size=64,sync=false),8,prefix_kernel!,top,sums,sums,Val(16),Val(false),workspaces[2])
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=16,shmem_size=4,sync=false),4096,addition_kernel!,output,top,workspaces[3])
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 4096;repeat=length(args)>1 ? parse(Int,args[2]) : 10;n==4096||Base.error("Named4096fixture only")
 input=Float32[mod(i*17,13)+1 for i in 0:n-1];expected=zeros(Float64,n);for i in 2:n;expected[i]=expected[i-1]+input[i-1];end
 di=CuArray(input);output=CUDA.zeros(Float32,n);sums=CUDA.zeros(Float32,16);top=CUDA.zeros(Float32,16);workspaces=Tuple(CUDA.zeros(Float32,n) for n in (256,16,1))
 pipeline!(output,di,sums,top,workspaces);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns();for _ in 1:repeat;pipeline!(output,di,sums,top,workspaces);end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
  emit(trial,seconds,Float64.(Array(output)),expected,repeat;case_times=[seconds])
 end
end
main(ARGS)
