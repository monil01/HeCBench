using CUDA
@inline function merge_values(count,mean,m2,newcount,newmean,newm2)
 factor=1f0/Float32(max(Int32(1),count+newcount));delta=mean-newmean
 mean=(newmean*Float32(newcount)+mean*Float32(count))*factor
 m2+=newm2+delta*delta*Float32(newcount)*Float32(count)*factor
 return count+newcount,mean,m2
end
@inline function put_state!(shared,i,count,mean,m2)
 shared[i]=reinterpret(UInt32,mean);shared[512+i]=reinterpret(UInt32,m2);shared[1024+i]=reinterpret(UInt32,count)
 return nothing
end
@inline function read_state(shared,i)
 return reinterpret(Int32,shared[1024+i]),reinterpret(Float32,shared[i]),reinterpret(Float32,shared[512+i])
end
function welford_kernel!(input,means,vars,bs,fs,ss,workspace)
 t=Int(threadIdx().x)-1;fid=Int(blockIdx().x)-1;tx=mod(t,32);ty=div(t,32)
 shared=CuStaticSharedArray(UInt32,1536)
 count=Int32(0);mean=0f0;m2=0f0
 batch=ty
 while batch<bs
  offset=tx
  while offset<ss
   count+=Int32(1);value=input[batch*ss*fs+fid*ss+offset+1];delta=value-mean;mean+=delta/Float32(count);m2+=delta*(value-mean);offset+=32
  end
  batch+=16
 end
 put_state!(shared,t+1,count,mean,m2)
 CUDA.sync_threads()
 offset=16
 while offset>0
  if tx<offset
   c,a,b=read_state(shared,t+1);nc,na,nb=read_state(shared,t+offset+1);c,a,b=merge_values(c,a,b,nc,na,nb);put_state!(shared,t+1,c,a,b)
  end
  CUDA.sync_threads()
  offset=div(offset,2)
 end
 if tx==0
  c,a,b=read_state(shared,t+1);put_state!(shared,ty+1,c,a,b)
 end
 CUDA.sync_threads()
 if t<32
  if t<16;c,a,b=read_state(shared,t+1);else;c=Int32(0);a=0f0;b=0f0;end
  put_state!(shared,t+1,c,a,b)
 end
 CUDA.sync_threads()
 offset=16
 while offset>0
  if t<offset
   c,a,b=read_state(shared,t+1);nc,na,nb=read_state(shared,t+offset+1);c,a,b=merge_values(c,a,b,nc,na,nb);put_state!(shared,t+1,c,a,b)
  end
  CUDA.sync_threads()
  offset=div(offset,2)
 end
 if t==0
  c,a,b=read_state(shared,1);means[fid+1]=a;vars[fid+1]=b/Float32(c)
 end
 return nothing
end
function launch(input,means,vars,bs,fs,ss,workspace)
 @cuda threads=512 blocks=fs welford_kernel!(input,means,vars,bs,fs,ss,nothing)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 bs=16;ss=64;fs=8;repeat=length(args)>0 ? parse(Int,args[1]) : 10
 input=Float32[Float32(mod(i*17,1024))/1024f0 for i in 0:bs*fs*ss-1];expectedmean=Float64[];expectedvar=Float64[]
 for f in 0:fs-1
  values=Float64[input[b*ss*fs+f*ss+s+1] for b in 0:bs-1 for s in 0:ss-1];mean=sum(values)/length(values);push!(expectedmean,mean);push!(expectedvar,sum((values.-mean).^2)/length(values))
 end
 expected=vcat(expectedmean,expectedvar);di=CuArray(input);means=CUDA.zeros(Float32,fs);vars=CUDA.zeros(Float32,fs);workspace=nothing;launch(di,means,vars,bs,fs,ss,workspace);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns();for _ in 1:repeat;launch(di,means,vars,bs,fs,ss,workspace);end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
  emit(trial,seconds,Float64.(vcat(Array(means),Array(vars))),expected,repeat;atol=1e-3,case_times=[seconds])
 end
end
main(ARGS)
