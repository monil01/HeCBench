using CUDA,Printf
include("study_device.jl")
function seam_run!(ops,state,M,indices,ref,seam,index,offset,W,H,S,mode)
 p,ps,L,U,R,LS,US,RS=state
 for step in Int32(0):S-Int32(1)
  cw=W-step
  if mode==0;ops.cost(p,L,U,R,cw);end
  if mode<2;ops.dp(L,U,R,M,cw);copyto!(indices,1,ref,1,Int(cw));ops.red(M,indices,cw,W*(H-1));ops.back(M,indices,seam,cw)
  else
   ops.approxsetup(p,index,offset,M,cw);st=Int32(1);ny=H÷Int32(2)
   while ny>0;ops.approxsum(offset,M,cw,st,ny);ny÷=Int32(2);st*=Int32(2);end
   copyto!(indices,1,ref,1,Int(cw));ops.red(M,indices,cw,Int32(0));ops.approxback(index,indices,seam)
  end
  ops.remove(p,ps,seam,cw);p,ps=ps,p
  if mode==1;ops.update(p,L,U,R,LS,US,RS,seam,cw);L,LS=LS,L;U,US=US,U;R,RS=RS,R;end
 end
 (p,ps,L,U,R,LS,US,RS)
end
function seam_reset!(ops,state,input,W,mode)
 p,ps,L,U,R,LS,US,RS=state;copyto!(p,input);if mode==1;ops.cost(p,L,U,R,W);end;CUDA.device_synchronize(blocking=true);nothing
end
function seam_collect(state,M,seam,index,offset,input,W,H,S,mode)
 p,ps,L,U,R,LS,US,RS=state;cw=W-S;out=vcat(input,UInt32[W,H,S,mode,1,cw+1,cw]);crop(a,rows,cols)=vec(reshape(Array(a),Int(W),Int(H))[1:Int(cols),1:Int(rows)])
 append!(out,crop(p,H,cw).&UInt32(0xffffff));append!(out,UInt32.(Array(seam)))
 if mode<2;append!(out,UInt32.(crop(M,H,cw+1)));costw=mode==1 ? cw : cw+1;for a in (L,U,R);append!(out,UInt32.(crop(a,H,costw)));end
 else;for a in (M,index,offset);append!(out,UInt32.(crop(a,H-1,cw+1)));end;end
 out
end
function seam_study(factory)
 W=Int32(128);H=Int32(128);S=Int32(100);mode=Int32(parse(Int,ARGS[1]));input=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"study_pixels_u32.bin"))));expected=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"study_expected_mode$(mode)_u32.bin"))));N=Int(W)*H
 p=CuArray(input);ps=similar(p);L=CuArray{Int16}(undef,N);U=similar(L);R=similar(L);LS=similar(L);US=similar(L);RS=similar(L);M=CuArray{Int32}(undef,N);indices=CuArray{Int32}(undef,W);ref=CuArray(collect(Int32(0):W-Int32(1)));seam=CuArray{Int32}(undef,H);index=similar(M);offset=similar(M);state=(p,ps,L,U,R,LS,US,RS);ops=factory(W,H,S)
 seam_reset!(ops,state,input,W,mode);state=seam_run!(ops,state,M,indices,ref,seam,index,offset,W,H,S,mode);CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  seam_reset!(ops,state,input,W,mode);start=time_ns();state=seam_run!(ops,state,M,indices,ref,seam,index,offset,W,H,S,mode);CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9
  out=seam_collect(state,M,seam,index,offset,input,W,H,S,mode);ok=out==expected;allok&=ok;err=abs.(Int64.(out).-Int64.(expected));ma=maximum(err);mr=maximum(Float64.(err)./max.(Float64.(expected),1e-300));path=joinpath(ENV["STUDY_OUTPUT_DIR"],"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%.12g,\"error_metric_domain\":\"exact_RGB_lastseam_defined_DP_cost_or_greedy_maps_source_behavior_not_minimum_energy_guarantee\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"seams_removed\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(out)-length(input)-7,ma,mr,path,length(out))
 end
 allok||error("source-behaviorRGB/seam/state CPU mismatch");println("PASS")
end
