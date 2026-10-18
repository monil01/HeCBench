function study(allocate,launch)
 f=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];i=1;S=-8192.25f0
 while i<=length(f)
  n,repeat=Int.(f[i:i+1]);i+=2;input=f[i:i+4n-1];i+=4n;h=vcat(fill(S,4),input,fill(S,4));a=allocate(Float32,4n+8);copyto!(a,h);v=reinterpret(F4,view(a,5:4n+4));out=allocate(Float32,n+8);fill!(out,S);push!(states,(n,repeat,h,a,v,out,view(out,5:n+4),input))
 end
 function run(c)
  n,repeat,h,a,v,out,ov,input=c
  for r in 1:repeat;launch(v,ov,Int32(n));end
  CUDA.synchronize(CUDA.context())
 end
 for c in states;run(c);end
 for trial in 1:5
  payload=copy(f);elapsed=0.;ok=true;oi=1;err=0.
  for c in states
   n,repeat,h,a,v,out,ov,input=c;fill!(out,S);CUDA.synchronize(CUDA.context());t=time_ns();run(c);elapsed+=(time_ns()-t)*1e-9/repeat;r=Array(out);back=Array(a);got=r[5:end-4];ref=oracle[oi:oi+n-1];oi+=n;errors=abs.(Float64.(got).-ref);err=max(err,maximum(errors));ok&=all(isfinite,got)&&all(errors.<=50.)&&reinterpret(UInt32,back)==reinterpret(UInt32,h)&&all(x==S for x in r[1:4])&&all(x==S for x in r[end-3:end])
   for j in 1:n
    if input[4j-3]==input[4j-1]&&input[4j-2]==input[4j];ok&=got[j]==0f0;end
   end
   append!(payload,back);append!(payload,r)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",err,",\"timing_region\":\"sum2_geodesic_10launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullGeodesic50mBudgetExactCoincidenceInputsGuards")
 end
end
