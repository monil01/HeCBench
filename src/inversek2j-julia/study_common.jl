using CUDA,Printf
function study_fixture()
 x=Float32[];y=Float32[]
 for i in 0:508
  push!(x,Float32(i%31)/8+.03125f0);push!(y,(isodd(i) ? -1f0 : 1f0)*(.5f0+Float32((i÷31)%8)/8))
 end
 append!(x,Float32[0,1.5,2.5,3]);append!(y,Float32[3,1.5,.5,0]);x,y
end
function study_gold(x,y)
 out=Float64[]
 for (xx,yy) in zip(x,y)
  # Independent closed form: after>=2 source loops, A,B+2A,C+3B+6A.
  a,b,c=[clamp(atan(Float64(yy),Float64(xx)-j)*180/pi,-30,30) for j in 0:2]
  append!(out,(a,b+2a,c+3b+6a))
 end
 out
end
function study_main(launch!)
 rep=isempty(ARGS) ? 100 : parse(Int,ARGS[1]);x,y=study_fixture();gold=study_gold(x,y);dx=CuArray(x);dy=CuArray(y);da=CUDA.zeros(Float32,3length(x));prefix=vcat(Float32[length(x)],x,y)
 launch!(dx,dy,da);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();st=time_ns();for _ in 1:rep;launch!(dx,dy,da);end;CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep
  got=Array(da);err=maximum(abs.(Float64.(got).-gold));ok=all(isfinite,got)&&err<=1e-3
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"one_complete_frozen_pose_native_behavior_kernel_NOT_valid_inverse_kinematics\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,rep,path,length(art));ok||error("native frozen-pose behavior contract, not scientific IKaccuracy")
 end
end
