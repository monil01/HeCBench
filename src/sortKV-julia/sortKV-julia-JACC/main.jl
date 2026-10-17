using CUDA,JACC
function bitonic_stage!(index,keys,values,n,k,j)
 i=index-1;partner=xor(i,j)
 if i<n&&partner>i&&partner<n
  left=keys[i+1];right=keys[partner+1];ascending=(i&k)==0
  if (ascending&&left>right)||(!ascending&&left<right)
   value=values[i+1];keys[i+1]=right;keys[partner+1]=left;values[i+1]=values[partner+1];values[partner+1]=value
  end
 end
end
function sort_device!(keys,values,n)
 k=2
 while k<=n
  j=div(k,2)
  while j>0
   JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,bitonic_stage!,keys,values,n,Int32(k),Int32(j))
   j=div(j,2)
  end
  k*=2
 end
 return keys,values
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 1024;repeat=length(args)>1 ? parse(Int,args[2]) : 3;ispow2(n)||Base.error("poweroftwo")
 types=(UInt8,Int16,Int32,Int64);keys=Int32[mod(i*2053,n) for i in 0:n-1];inputs=[T.(mod.(keys,256)) for T in types];dkeys=CUDA.zeros(Int32,n);dvalues=[CUDA.zeros(T,n) for T in types];hostkeys=zeros(Int32,n);hostvalues=[zeros(T,n) for T in types]
 function operation!(case)
  copyto!(dkeys,keys);copyto!(dvalues[case],inputs[case]);sk,sv=sort_device!(dkeys,dvalues[case],n);copyto!(hostkeys,sk);copyto!(hostvalues[case],sv);CUDA.synchronize()
 end
 for case in 1:4;operation!(case);end
 expected=reduce(vcat,[vcat(Float64.(0:n-1),Float64.(mod.(0:n-1,256))) for _ in 1:4])
 for trial in 1:5
  result=Float64[];times=Float64[]
  for case in 1:4
   CUDA.synchronize();start=time_ns();for _ in 1:repeat;operation!(case);end;CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Float64.(hostkeys));append!(result,Float64.(hostvalues[case]))
  end
  emit(trial,sum(times),result,expected,repeat;case_times=times)
 end
end
main(ARGS)
