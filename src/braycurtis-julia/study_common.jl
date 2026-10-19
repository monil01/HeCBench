using CUDA,Printf
include(joinpath(@__DIR__,"study_cases.jl"))
function bray_read(name,::Type{T},n)where T
 bytes=read(joinpath(@__DIR__,name));length(bytes)==n*sizeof(T)||error("asset bytes $name");collect(reinterpret(T,bytes))
end
function forward_batch!(launch,x)
 for _ in 1:x.c.repeats;launch(1,x);end
 CUDA.device_synchronize(blocking=true)
end
function backward_batch!(launch,x)
 for _ in 1:x.c.repeats
  launch(2,x);launch(3,x);launch(4,x)
 end
 CUDA.device_synchronize(blocking=true)
end
function bray_study(launch,observer;allocate=CuArray)
 works=map(BRAY_CASES)do c
  T=c.dtype==32 ? Float32 : Float64;a=allocate(bray_read("case$(c.id)_x1.bin",T,c.na+16));b=allocate(bray_read("case$(c.id)_x2.bin",T,c.nb+16));g=allocate(bray_read("case$(c.id)_go.bin",T,c.nq+16));reset(n)=[fill(T(91.25),8);fill(T(NaN),n);fill(T(91.25),8)]
  ro=reset(Int(c.nq));r1=reset(Int(c.na));r2=reset(Int(c.nb));o=allocate(ro);sd=allocate(ro);ss=allocate(ro);g1=allocate(r1);g2=allocate(r2);echo=allocate(zeros(Int32,5))
  (c=c,a=a,b=b,g=g,o=o,sd=sd,ss=ss,g1=g1,g2=g2,params=echo,ro=ro,r1=r1,r2=r2,args=(c,view(a,9:length(a)-8),view(b,9:length(b)-8),view(g,9:length(g)-8),view(o,9:length(o)-8),view(sd,9:length(sd)-8),view(ss,9:length(ss)-8),view(g1,9:length(g1)-8),view(g2,9:length(g2)-8)))
 end
 expected=bray_read("oracle_f64.bin",Float64,BRAY_COUNT);at=bray_read("oracle_atol_f64.bin",Float64,BRAY_COUNT);rt=bray_read("oracle_rtol_f64.bin",Float64,BRAY_COUNT)
 # Warm the exact forward andbackward host loops atbothprecisions/allshapes.
 for x in works;forward_batch!(launch,x);backward_batch!(launch,x);end
 for trial in 1:5
  pre=Float64[];outs=Float64[];secs=0.0
  for x in works
   c=x.c;for dest in (x.o,x.sd,x.ss);copyto!(dest,x.ro);end;copyto!(x.g1,x.r1);copyto!(x.g2,x.r2)
   before=(Array(x.a),Array(x.b),Array(x.g));observer(x);echo=Array(x.params);CUDA.device_synchronize(blocking=true)
   t=time_ns();forward_batch!(launch,x);secs+=(time_ns()-t)*1e-9/c.repeats
   t=time_ns();backward_batch!(launch,x);secs+=(time_ns()-t)*1e-9/c.repeats
   append!(pre,Float64[c.dtype,c.b,c.m,c.n,c.k,c.repeats]);append!(pre,Float64.(echo));for v in before;append!(pre,Float64.(v));end;for v in (x.a,x.b,x.g);append!(pre,Float64.(Array(v)));end
   for v in (x.o,x.sd,x.ss,x.g1,x.g2);append!(outs,Float64.(Array(v)));end
  end
  length(pre)==BRAY_PREFIX||error("prefixcount");append!(pre,outs);length(pre)==BRAY_COUNT||error("outputcount");ok=true;ma=0.0;mr=0.0
  for i in eachindex(pre)
   d=abs(pre[i]-expected[i]);ok &= isfinite(pre[i])&&d<=at[i]+rt[i]*abs(expected[i]);ma=max(ma,d);mr=max(mr,d/max(abs(expected[i]),1e-300))
  end
  path=joinpath(ENV["STUDY_OUTPUT_DIR"],"bray_trial$(trial).bin");write(path,pre)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"jit_excluded\":true,\"timing_region\":\"sum6_source_forward_plus3kernel_backward_means_R10\",\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(pre),ma,mr,path,length(pre));flush(stdout);ok||error("independent fullgradient/distance/guards criterion failed")
 end
end
