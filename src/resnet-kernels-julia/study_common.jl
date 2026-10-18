using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
const ST_SHAPES=((128,128),(256,256),(512,128),(128,512),(1024,256),(256,1024))
function prepare()
 frames=Any[];pos=1
 for (mode,(ci,co)) in enumerate(ST_SHAPES)
  win=mode<=2;sz=(win ? 290*ci : 196*ci,win ? 36*ci*co : ci*co,co,co)
  arrays=map(sz) do n
   a=CuArray(ST_INPUT[pos:pos+n-1]);pos+=n;a
  end
  n=win ? 576*ci : 0
  t=CUDA.fill(-12345f0,n+17);ip=CUDA.fill(-12345f0,n+17);o=CUDA.fill(-12345f0,(win ? 256 : 196)*co+17)
  if win;CUDA.fill!(view(o,1:256*co),0f0);end
  push!(frames,(arrays...,t,ip,o,Int32(ci),Int32(co),Int32(mode-1)))
 end
 @assert pos==length(ST_INPUT)+1
 frames
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);frames=prepare()
 for f in frames;launch!(f);CUDA.synchronize();end
 for trial in 1:5
  times=Float64[];got=Float32[];gct=0.;gcp=0
  for f in frames
   CUDA.synchronize();gb=Base.gc_num();t=time_ns()
   for _ in 1:rep;launch!(f);end
   CUDA.synchronize();push!(times,(time_ns()-t)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   if f[end]<2;append!(got,Array(f[5]));append!(got,Array(f[6]));end
   append!(got,Array(f[7]))
  end
  ok=true;ma=0.;mr=0.
  for i in eachindex(got)
   e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);mr=max(mr,e/max(abs(ST_GOLD[i]),1e-12));ok&=isfinite(got[i])&&(ST_GOLD[i]==-12345 ? got[i]==-12345 : e<=1e-4)
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum6_source_convolution_paths_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
