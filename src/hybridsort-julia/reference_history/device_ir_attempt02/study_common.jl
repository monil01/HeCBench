
using Printf
const BASE=dirname(@__FILE__);const N=16384;const CAP=N+4096;const G=32
readf(n)=collect(reinterpret(Float32,read(joinpath(BASE,n))))
# CPU pivot algorithm retained verbatim in control/math; histogram is intentionally mutated.
function pivots!(hist,n,lo,hi)
 width=(hi-lo)/1024f0;per=Float32(n)/1024f0;at=lo;ends=lo+width;need=per;p=Float32[];sizehint!(p,1024)
 for i=1:1024
  if i==1024
   if !(length(p)<1024);push!(p,at+(need/hist[i])*width);end
   break
  end
  while hist[i]>need
   length(p)<1024 || error("source pivot overflow")
   push!(p,at+(need/hist[i])*width);at+=(need/hist[i])*width;hist[i]-=need;need=per
  end
  need-=hist[i];at=ends;ends+=width
 end
 while length(p)<1024;push!(p,p[end]);end
 p
end
function pipeline(input,output,lo,hi)
 # Same original end-to-end ROI: allocation, transfers, CPU pivots/alignment, all device stages.
 din=CuArray(input[1:CAP]);off=CUDA.zeros(UInt32,1024);dummy=off
 idx=CUDA.zeros(Int32,N);pre=CuArray{UInt32}(undef,cld(N,4096)*1024)
 args=(din,din,idx,pre,off,din,idx,off,off,N,lo,hi,0,0,0)
 launch(Val(1),96,64,args...)
 hist=Float32.(Array(off));p=pivots!(hist,N,lo,hi);dp=CuArray(p)
 launch(Val(2),32,cld(N,4096),din,din,idx,pre,off,dp,idx,off,off,N,lo,hi,0,0,0)
 launch(Val(3),128,8,din,din,idx,pre,off,dp,idx,off,off,N,lo,hi,0,0,cld(N,4096))
 counts=Array(off);original=Vector{UInt32}(undef,1025);original[1]=0;nulls=Vector{UInt32}(undef,1024);sizes=Vector{Int32}(undef,1024);offsets=Vector{UInt32}(undef,1024)
 z=UInt32(0)
 for i=1:1024
  original[i+1]=original[i]+counts[i];nulls[i]=UInt32(mod(-Int(counts[i]),4));sizes[i]=Int32((counts[i]+nulls[i])÷4);offsets[i]=z;z+=counts[i]+nulls[i]
 end
 copyto!(off,offsets);bucket=CuArray(output[1:CAP])
 launch(Val(4),32,cld(N,4096),din,bucket,idx,pre,off,dp,idx,off,off,N,lo,hi,0,0,0)
 CUDA.device_synchronize(blocking=true);copyto!(view(output,1:CAP),Array(bucket))
 # Original merge phase allocates a second device copy and transfers bucket output through host.
 newsize=sum(Int.(sizes))*4;starts=Int32[0;cumsum(sizes)];largest=maximum(sizes)*4
 a=CuArray(output[1:newsize]);b=CuArray{Float32}(undef,newsize);ds=CuArray(starts)
 launch(Val(5),256,cld(newsize÷4,256),a,b,idx,pre,off,dp,ds,off,off,N,lo,hi,0,0,newsize÷4)
 nr=2
 while true
  tpd=cld(largest,nr*4);needed=tpd*1024;th=208;bl=cld(needed,th)
  if bl<8;bl=8;th=cld(needed,bl);end
  a,b=b,a
  launch(Val(6),th,bl,a,b,idx,pre,off,dp,ds,off,off,N,lo,hi,tpd,nr,newsize÷4)
  nr*=2
  tpd==1 && break
 end
 df=CuArray(original);dn=CuArray(nulls)
 launch(Val(7),256,(cld(largest,256),1024),b,a,idx,pre,off,dp,ds,df,dn,N,lo,hi,0,0,newsize÷4)
 CUDA.device_synchronize(blocking=true);copyto!(view(output,1:newsize),Array(a))
 for d in (din,off,idx,pre,dp,bucket,a,b,ds,df,dn);CUDA.unsafe_free!(d);end
 (sizes,nulls,original)
end
function main()
 inputs=[readf("study_case$(c)_f32.bin") for c=1:2];oracles=[readf("oracle_case$(c)_f32.bin") for c=1:2]
 for c=1:2
  input=vcat(inputs[c],zeros(Float32,4096),fill(12345.5f0,G));output=vcat(zeros(Float32,CAP),fill(12345.5f0,G))
  pipeline(input,output,minimum(inputs[c]),maximum(inputs[c]))
 end
 for trial=1:5
  prefix=Float64[];outs=Float64[];guards=Float64[];total=0.;ok=true
  for c=1:2
   input=vcat(inputs[c],zeros(Float32,4096),fill(12345.5f0,G));output=vcat(zeros(Float32,CAP),fill(12345.5f0,G));lo=minimum(inputs[c]);hi=maximum(inputs[c])
   CUDA.device_synchronize(blocking=true);t=time_ns();sizes,nulls,original=pipeline(input,output,lo,hi);CUDA.device_synchronize(blocking=true);total+=(time_ns()-t)*1e-9
   append!(prefix,Float64.(reinterpret(UInt32,inputs[c])));append!(prefix,[N,1024,96,64,32,128,256,208,c])
   append!(outs,Float64.(output[1:N]));append!(outs,Float64.(sizes));append!(outs,Float64.(nulls));append!(outs,Float64.(original))
   for i=CAP+1:CAP+G;push!(guards,input[i],output[i]);end
   ok &= output[1:N]==oracles[c] && all(0 .<= nulls .<= 3) && all(4 .* sizes.-nulls .== diff(original)) && original[end]==N
  end
  ok &= all(==(12345.5),guards);allout=vcat(prefix,outs,guards);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR","/tmp"),"trial$(trial).bin");write(path,reinterpret(UInt8,allout))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":39042,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum2_full_histogram_cpu_pivot_bucket_merge_pack_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,total,ok ? "true" : "false",path,length(allout));flush(stdout);ok || error("CPU fullsort/count/guard check failed")
 end
end
