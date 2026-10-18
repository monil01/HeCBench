using CUDA,Printf
function pns_study(factory)
 n=Int32(1024);s=Int32(312);t=4;seed=Int32(5489);Q=Int(2n*n);W=Q÷4+Q
 expectedmarks=read(joinpath(@__DIR__,"study_expected_marks_u8.bin"));expectedconflicts=collect(reinterpret(Int32,read(joinpath(@__DIR__,"study_expected_conflicts_i32.bin"))));expectedrewards=collect(reinterpret(Float64,read(joinpath(@__DIR__,"study_expected_rewards_f64.bin"))));g=CUDA.zeros(UInt32,W*t);v=CUDA.zeros(Float32,t);m=CUDA.zeros(Int32,t);launch=factory(g,v,m,n,s,seed);fill!(g,0);CUDA.device_synchronize(blocking=true);launch();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  fill!(g,0);CUDA.device_synchronize(blocking=true);start=time_ns();launch();CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9;state=Array(g);vars=Array(v);maxs=Array(m);out=Float32[n,s,t,seed];ok=true;ma=0.;mr=0.
  for b in 0:t-1
   marks=reinterpret(UInt8,state)[b*W*4+1:b*W*4+Q];conflicts=reinterpret(Int32,state)[b*W+Q÷4+1:b*W+Q÷4+Q];target=expectedmarks[b*Q+1:(b+1)*Q];ctarget=expectedconflicts[b*Q+1:(b+1)*Q];diff=abs.(Int.(marks).-Int.(target));cdiff=abs.(Int.(conflicts).-Int.(ctarget));ma=max(ma,maximum(diff),maximum(cdiff),abs(Float64(vars[b+1])-expectedrewards[2b+1]),abs(Float64(maxs[b+1])-expectedrewards[2b+2]));mr=max(mr,maximum(diff./max.(Int.(target),1)),maximum(cdiff./max.(abs.(Int.(ctarget)),1)),abs(Float64(vars[b+1])-expectedrewards[2b+1])/max(abs(expectedrewards[2b+1]),1e-12),abs(Float64(maxs[b+1])-expectedrewards[2b+2])/max(abs(expectedrewards[2b+2]),1e-12));ok&=marks==target&&conflicts==ctarget&&Float64(vars[b+1])==expectedrewards[2b+1]&&Float64(maxs[b+1])==expectedrewards[2b+2]&&sum(Int,marks)==Q&&maximum(marks)==maxs[b+1];append!(out,Float32.(marks));append!(out,Float32.(conflicts));push!(out,vars[b+1],Float32(maxs[b+1]))
  end
  allok&=ok;length(out)==16777228||error("artifactshape");path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(out),ma,mr,path,length(out))
 end
 allok||error("Petri full marking/conflict/reward mismatch");println("PASS")
end
