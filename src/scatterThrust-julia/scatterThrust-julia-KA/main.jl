using CUDA,KernelAbstractions
@kernel function scatter_kernel!(out,src,idx,n)
 i=@index(Global,Linear)
 if i<=n;out[idx[i]+1]=src[i];end
end
value_for(::Type{Int8}, i::Int64) = reinterpret(Int8, UInt8(i % 256))
value_for(::Type{Int16}, i::Int64) = reinterpret(Int16, UInt16(i % 65536))
value_for(::Type{Int32}, i::Int64) = Int32(i)
value_for(::Type{Int64}, i::Int64) = i
value_for(::Type{Float32}, i::Int64) = Float32(i)
value_for(::Type{Float64}, i::Int64) = Float64(i)

function launch(out,src,indices,n)
 scatter_kernel!(get_backend(src),256)(out,src,indices,n;ndrange=n)
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
 types=(Int8,Int16,Int32,Int64,Float32,Float64);inputs=[CuArray([value_for(T,Int64(i)) for i in 0:n-1]) for T in types];outputs=[CUDA.zeros(T,n) for T in types];indices=CuArray(Int64.(n-1:-1:0));expected=reduce(vcat,[Float64[value_for(T,Int64(i)) for i in n-1:-1:0] for T in types])
 for case in 1:6;launch(outputs[case],inputs[case],indices,n);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for case in 1:6
   out=outputs[case];input=inputs[case];seconds=0.
   for _ in 1:repeat
    fill!(out,zero(eltype(out)));CUDA.synchronize();start=time_ns();launch(out,input,indices,n);CUDA.synchronize();seconds+=(time_ns()-start)*1e-9/repeat
   end
   push!(times,seconds);append!(result,Float64.(Array(out)))
  end
  emit(trial,sum(times),result,expected,repeat;case_times=times)
 end
end
main(ARGS)
