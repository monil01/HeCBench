using CUDA,Printf
include(joinpath(@__DIR__,"study_params.jl"))
bsw_guarded(a,g)=[fill(eltype(a)(g),16);a;fill(eltype(a)(g),16)]
function bsw_study(launch;allocate=CuArray)
 wrap(a,g)=allocate(bsw_guarded(a,g))
 a=wrap(zeros(UInt8,NA),211);b=wrap(zeros(UInt8,NB),211);oa=wrap(zeros(UInt32,N),24681357);ob=wrap(zeros(UInt32,N),24681357);sc=wrap(SCORING,-12345);en=wrap(ENCODING,-12345);outputs=[wrap(fill(Int16(-12345),N),-12345) for _ in 1:5]
 av(x)=view(x,17:length(x)-16)
 arguments=(av(a),av(b),av(oa),av(ob),(av(x) for x in outputs)...,Int16(-6),Int16(-1),av(sc),av(en))
 refs=String[];queries=String[];prevA=0;prevB=0
 for k in 1:N
  push!(refs,String(copy(SEQ_A[prevA+1:Int(OFF_A[k])])));push!(queries,String(copy(SEQ_B[prevB+1:Int(OFF_B[k])])));prevA=Int(OFF_A[k]);prevB=Int(OFF_B[k])
 end
 packA=zeros(UInt8,NA);packB=zeros(UInt8,NB);offsetA=zeros(UInt32,N);offsetB=zeros(UInt32,N);endA=zeros(Int16,N);endB=zeros(Int16,N);startA=zeros(Int16,N);startB=zeros(Int16,N);top=zeros(Int16,N)
 function reset!()
  copyto!(a,bsw_guarded(zeros(UInt8,NA),211));copyto!(b,bsw_guarded(zeros(UInt8,NB),211));copyto!(oa,bsw_guarded(zeros(UInt32,N),24681357));copyto!(ob,bsw_guarded(zeros(UInt32,N),24681357))
  for x in outputs;copyto!(x,fill(Int16(-12345),N+32));end
 end
 function capture_inputs!(p)
  for x in (a,b,oa,ob,sc,en);append!(p,Int32.(Array(x)));end
 end
 function capture_outputs!(p)
  for x in outputs;append!(p,Int32.(Array(x)));end
 end
 function pipeline!()
  sequencesA=copy(refs);sequencesB=copy(queries);runningA=0;runningB=0
  for k in 1:N
   runningA+=ncodeunits(sequencesA[k]);offsetA[k]=UInt32(runningA);runningB+=ncodeunits(sequencesB[k]);offsetB[k]=UInt32(runningB)
  end
  posA=0;posB=0
  for k in 1:N
   ca=codeunits(sequencesA[k]);cb=codeunits(sequencesB[k]);copyto!(packA,posA+1,ca,1,length(ca));copyto!(packB,posB+1,cb,1,length(cb));posA+=length(ca);posB+=length(cb)
  end
  copyto!(av(oa),offsetA);copyto!(av(ob),offsetB);copyto!(av(a),packA);copyto!(av(b),packB)
  launch(false,FORWARD,arguments...)
  copyto!(endA,av(outputs[2]));copyto!(endB,av(outputs[4]));CUDA.device_synchronize(blocking=true)
  reverse=Int(min(maximum(endA),maximum(endB)));launch(true,reverse,arguments...)
  copyto!(startA,av(outputs[1]));copyto!(startB,av(outputs[3]));copyto!(top,av(outputs[5]));CUDA.device_synchronize(blocking=true)
  reverse
 end
 expected=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle_i32.bin"))));length(expected)==COUNT||error("CPUoracle extent")
 # Warm exact hostprefix/packing/transfer/kernel/hostfeedback/reverse/copy loop.
 reset!();pipeline!();CUDA.device_synchronize(blocking=true)
 for trial in 1:5
  reset!();before=Int32[];initial=Int32[];capture_inputs!(before);capture_outputs!(initial);CUDA.device_synchronize(blocking=true)
  t=time_ns();reverse=pipeline!();CUDA.device_synchronize(blocking=true);secs=(time_ns()-t)*1e-9
  actual=Int32[N,160,128,-6,-1,FORWARD,reverse];append!(actual,before);capture_inputs!(actual);append!(actual,Int32.(endA));append!(actual,Int32.(endB));append!(actual,initial);length(actual)==PREFIX||error("prefix");capture_outputs!(actual);length(actual)==COUNT||error("capturecount")
  ok=actual==expected
  for k in 1:N
   ok &= startA[k]==actual[PREFIX+16+k]&&endA[k]==actual[PREFIX+(N+32)+16+k]&&startB[k]==actual[PREFIX+2*(N+32)+16+k]&&endB[k]==actual[PREFIX+3*(N+32)+16+k]&&top[k]==actual[PREFIX+4*(N+32)+16+k]
  end
  err=maximum(abs(Int64(actual[i])-Int64(expected[i])) for i in eachindex(actual));path=joinpath(ENV["STUDY_OUTPUT_DIR"],"bsw_trial$(trial).bin");write(path,actual)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"named_source_its1_hostpacking_H2D_forward_host_end_feedback_reverse_D2H\",\"internal_repeats\":1,\"iterations\":1,\"forward_threads\":%d,\"reverse_threads\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(actual),err,FORWARD,reverse,path,length(actual));flush(stdout);ok||error("independent full65proteinDP/coordinates/actualinputs/guards checks")
 end
end
