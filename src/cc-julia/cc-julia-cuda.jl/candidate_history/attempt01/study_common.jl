using CUDA,Printf,TOML
const CC_PARENT=@__DIR__
const CC_CASES=[(id=0,n=Int32(1),e=Int32(0),repeats=100,blocks=640,ic=34,ac=33),(id=1,n=Int32(65),e=Int32(124),repeats=100,blocks=640,ic=98,ac=156),(id=2,n=Int32(4099),e=Int32(12284),repeats=100,blocks=640,ic=4132,ac=12316)]
const CC_GUARD=Int32(-123456789);const CC_PREFIX=33559;const CC_COUNT=37916
function cc_read(name,count)
 bytes=read(joinpath(CC_PARENT,name));length(bytes)==count*4||error("fixture size $name");collect(reinterpret(Int32,bytes))
end
function cc_batch!(pipeline,x)
 for _ in 1:x.c.repeats;pipeline(x);end
 CUDA.device_synchronize(blocking=true)
end
function cc_study(pipeline,observer;allocate=CuArray)
 works=map(CC_CASES)do c
  n=Int(c.n);a=allocate(cc_read("case$(c.id)_idx_i32.bin",c.ic));b=allocate(cc_read("case$(c.id)_adj_i32.bin",c.ac));g=fill(CC_GUARD,n+32);s=allocate(g);w=allocate(g);ctr=allocate(zeros(Int32,4));params=allocate(zeros(Int32,3))
  (c=c,a=a,b=b,s=s,w=w,idx=view(a,17:length(a)-16),adj=view(b,17:length(b)-16),stat=view(s,17:n+16),wl=view(w,17:n+16),ctr=ctr,params=params,reset=g)
 end
 expected=cc_read("oracle_i32.bin",CC_COUNT)
 # Warm complete host batch and every source stage at each shape.
 for x in works;cc_batch!(pipeline,x);end
 # Actual untimed full stage traces are produced by target entrypoint diagnostic, separately from ROI records.
 for trial in 1:5
  pre=Int32[];out=Int32[];times=Float64[]
  for x in works
   c=x.c;n=Int(c.n);copyto!(x.s,x.reset);copyto!(x.w,x.reset);ib=Array(x.a);ab=Array(x.b)
   observer(x);params=Array(x.params);CUDA.device_synchronize(blocking=true)
   t=time_ns();cc_batch!(pipeline,x);push!(times,(time_ns()-t)*1e-9/c.repeats)
   append!(pre,Int32[c.n,c.e,c.repeats,c.blocks]);append!(pre,params);append!(pre,ib);append!(pre,ab);append!(pre,Array(x.a));append!(pre,Array(x.b));append!(out,Array(x.s));ww=Array(x.w);append!(out,ww[1:16]);append!(out,ww[end-15:end])
  end
  length(pre)==CC_PREFIX||error("prefix count");append!(pre,out);length(pre)==CC_COUNT||error("artifact count");ok=pre==expected
  path=joinpath(ENV["STUDY_OUTPUT_DIR"],"cc_trial$(trial).bin");write(path,pre)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%s,\"max_rel_error\":%s,\"jit_excluded\":true,\"timing_region\":\"sum3_source5stage_pipeline_batch_means_R100\",\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(pre),ok ? "0" : "null",ok ? "0" : "null",path,length(pre));flush(stdout);ok||error("full BFS/CSR/guard oracle failure")
 end
end
