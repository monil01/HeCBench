using CUDA,KernelAbstractions
@inline function scalar_packed_dot(a::T,b::T) where T
 ua=reinterpret(UInt32,a);ub=reinterpret(UInt32,b);value=zero(T)
 for shift in (0,8,16,24)
  av=Int32((ua>>shift)&0xff);bv=Int32((ub>>shift)&0xff)
  if T==Int32
   av=av>=128 ? av-256 : av;bv=bv>=128 ? bv-256 : bv
  end
  value+=T(av*bv)
 end
 return value
end
@kernel function dot_kernel!(a,b,out::AbstractVector{T},ntasks,::Val{INTRINSIC},workspace) where {T,INTRINSIC}
 tid=@index(Local,Linear);block=@index(Group,Linear);index=@index(Global,Linear)-1
 shared=@localmem T (1024,)
 total=zero(T)
 if index<ntasks
  for word in 1:4
   offset=4index+word
   if INTRINSIC
    total=CUDA.CUDACore.dp4a(a[offset],b[offset],total)
   else
    total+=scalar_packed_dot(a[offset],b[offset])
   end
  end
 end
 shared[tid]=total
 @synchronize
 offset=512
 while offset>0
  if tid<=offset;shared[tid]+=shared[tid+offset];end
  @synchronize
  offset=div(offset,2)
 end
 if tid==1;out[block]=shared[1];end
end
function launch(a,b,out,n,mode,workspace)
 dot_kernel!(get_backend(a),1024)(a,b,out,div(n,4),mode,nothing;ndrange=div(n,4))
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||Base.error("numerical failure")
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 16384;repeat=length(args)>1 ? parse(Int,args[2]) : 10;n%4096==0||Base.error("shape")
 packed=UInt32[UInt32((i*17)%256)|(UInt32((i*31+3)%256)<<8)|(UInt32((i*47+7)%256)<<16)|(UInt32((i*61+11)%256)<<24) for i in 0:n-1]
 inputs=(CuArray(reinterpret(Int32,packed)),CuArray(packed));outputs=(CUDA.zeros(Int32,div(n,4096)),CUDA.zeros(UInt32,div(n,4096)))
 expected=Float64[]
 for data in (reinterpret(Int32,packed),packed)
  blockvalues=[sum(Float64(scalar_packed_dot(data[i],data[i])) for i in (block-1)*4096+1:block*4096) for block in 1:div(n,4096)]
  append!(expected,blockvalues);append!(expected,blockvalues)
 end
 for typecase in 1:2,mode in (Val(false),Val(true));launch(inputs[typecase],inputs[typecase],outputs[typecase],n,mode,nothing);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for typecase in 1:2,mode in (Val(false),Val(true))
   a=inputs[typecase];out=outputs[typecase];CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;launch(a,a,out,n,mode,nothing);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Float64.(Array(out)))
  end
  emit(trial,sum(times),result,expected,repeat;case_times=times)
 end
end
main(ARGS)
