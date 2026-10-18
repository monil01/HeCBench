using CUDA,Printf
const BFloat16=CUDA.CUDACore.BFloat16
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
function prepare()
 fs=Any[];pos=2
 for _ in 1:Int(ST_INPUT[1])
  kind,b,h,q,c=Int.(ST_INPUT[pos:pos+4]);scale=ST_INPUT[pos+5];n=Int(ST_INPUT[pos+6]);pos+=7;T=kind==0 ? Float16 : BFloat16
  src=CuArray(T.(ST_INPUT[pos:pos+n-1]));pos+=n;mask=CuArray(UInt8.(ST_INPUT[pos:pos+n-1]));pos+=n
  for pad in (0,1);push!(fs,(src,mask,CUDA.fill(T(-8),n+17),Vector{T}(undef,n+17),scale,Int32(b),Int32(h),Int32(q),Int32(c),Int32(pad)));end
 end
 @assert pos==length(ST_INPUT)+1;fs
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fs=prepare();for f in fs;launch!(f);CUDA.synchronize();end;got=Vector{Float32}(undef,length(ST_GOLD))
 for trial in 1:5
  times=Float64[];offset=1;gct=0.;gcp=0
  for f in fs
   CUDA.fill!(f[3],eltype(f[3])(-8));CUDA.synchronize();gb=Base.gc_num();t=time_ns();for _ in 1:rep;launch!(f);end;CUDA.synchronize();push!(times,(time_ns()-t)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   copyto!(f[4],f[3]);for x in f[4];got[offset]=Float32(x);offset+=1;end
  end
  @assert offset==length(got)+1;ok=true;ma=0.;mr=0.
  for i in eachindex(got);e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);mr=max(mr,e/max(abs(ST_GOLD[i]),1e-12));ok&=isfinite(got[i])&&(ST_GOLD[i]==-8 ? got[i]==-8 : e<=1e-3);end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum16_scaledmasked_FP16_BF16_paths\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
