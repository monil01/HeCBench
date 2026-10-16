using CUDA,KernelAbstractions
@kernel function scalar_kernel!(a,b,r,n)
 i=@index(Global,Linear)
 if i<=n;r[i]=max(a[i],b[i]);end
end
@kernel function packed_kernel!(a,b,r,n)
 i=@index(Global,Linear)
 if i<=n
  r[2i-1]=max(a[2i-1],b[2i-1]);r[2i]=max(a[2i],b[2i])
 end
end
function launch(case,a,b,r,n)
 if case==1
  packed_kernel!(get_backend(a),256)(a,b,r,n;ndrange=n)
 else
  scalar_kernel!(get_backend(a),256)(a,b,r,2n;ndrange=2n)
 end
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 4096;repeat=length(args)>1 ? parse(Int,args[2]) : 10
 a=Float16[Float16(mod(i*17,257)-128)+Float16(.125) for i in 0:2n-1];b=Float16[Float16(mod(i*31,251)-125)+Float16(.375) for i in 0:2n-1]
 expected=repeat_values=Float64.(max.(a,b));expected=vcat(expected,expected);da=CuArray(a);db=CuArray(b);dr=CUDA.zeros(Float16,2n)
 for case in 1:2;launch(case,da,db,dr,n);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for case in 1:2
   CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;launch(case,da,db,dr,n);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Float64.(Array(dr)))
  end
  emit(trial,sum(times),result,expected,repeat;case_times=times)
 end
end
main(ARGS)
