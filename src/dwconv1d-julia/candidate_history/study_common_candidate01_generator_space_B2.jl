const SENT=-8192.25f0
function study(allocate,launch)
 f=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))))
 oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))))
 states=[];i=1
 while i<=length(f)
  B,C,T=Int.(f[i:i+2]);eps=f[i+4];i+=5
  arr=[];host=[]
  for n in(C*T,B*C*T,B*C*T)
   h=vcat(fill(SENT,4),f[i:i+n-1],fill(SENT,4));i+=n;a=allocate(Float32,n+8);copyto!(a,h);push!(arr,a);push!(host,h)
  end
  for j in 1:3;a=allocate(Float32,B*C*T+8);fill!(a,SENT);push!(arr,a);end
  push!(states,(B,C,T,eps,arr,host,[view(z,5:length(z)-4)for z in arr]))
 end
 function run(c)
  B,C,T,eps,a,h,v=c
  launch(v,eps,B,C,T);CUDA.synchronize(CUDA.context())
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(f);elapsed=0.;ok=true;oi=1;err=0.
  for c in states
   B,C,T,eps,a,h,v=c;CUDA.synchronize(CUDA.context());t=time_ns();run(c);elapsed+=(time_ns()-t)*1e-9
   for j in 1:3;r=Array(a[j]);ok&=reinterpret(UInt32,r)==reinterpret(UInt32,h[j]);append!(out,r[1:4]);append!(out,r[end-3:end]);end
   for j in 4:6
    r=Array(a[j]);got=r[5:end-4];expected=oracle[oi:oi+length(got)-1];oi+=length(got);e=abs.(Float64.(got).-expected);err=max(err,maximum(e));ok&=all(e.<=2e-5.+2e-5.*abs.(expected))&&all(x==SENT for x in r[1:4])&&all(x==SENT for x in r[end-3:end]);append!(out,got);append!(out,r[1:4]);append!(out,r[end-3:end])
   end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum3_forward_backward_preallocated_kernel_pipeline_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullForwardBackwardOracleInputGuards")
 end
end
