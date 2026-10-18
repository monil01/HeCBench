using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
function prepare()
 frames=Any[];pos=2
 for _ in 1:Int(ST_INPUT[1])
  outer,inner,n=Int.(ST_INPUT[pos:pos+2]);pos+=3
  logits=CuArray(ST_INPUT[pos:pos+n-1]);pos+=n;targets=CuArray(ST_INPUT[pos:pos+n-1]);pos+=n
  push!(frames,(logits,targets,CUDA.fill(-12345f0,outer+17),Int32(outer),Int32(inner)))
 end
 @assert pos==length(ST_INPUT)+1
 frames
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);frames=prepare()
 for f in frames,mode in Int32(0):Int32(2);launch!(f,mode);CUDA.synchronize();end
 for trial in 1:5
  times=Float64[];got=Float32[];gct=0.;gcp=0
  for f in frames,mode in Int32(0):Int32(2)
   CUDA.fill!(f[3],-12345f0);CUDA.synchronize();gb=Base.gc_num();t=time_ns()
   for _ in 1:rep;launch!(f,mode);end
   CUDA.synchronize();push!(times,(time_ns()-t)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;append!(got,Array(f[3]))
  end
  ok=true;ma=0.;mr=0.
  for i in eachindex(got)
   e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);mr=max(mr,e/max(abs(ST_GOLD[i]),1e-12));ok&=isfinite(got[i])&&(ST_GOLD[i]==-12345 ? got[i]==-12345 : e<=1e-3)
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum15_independent_sigmoid_loss_kernel_paths\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
