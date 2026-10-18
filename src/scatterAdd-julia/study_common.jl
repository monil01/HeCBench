using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
function prepare()
 fs=Any[];pos=2
 for _ in 1:Int(ST_INPUT[1])
  n,o,v=Int.(ST_INPUT[pos:pos+2]);pos+=3;idx=CuArray(Int32.(ST_INPUT[pos:pos+n-1]));pos+=n;src=CuArray(ST_INPUT[pos:pos+n*v-1]);pos+=n*v;init=CuArray(ST_INPUT[pos:pos+o*v-1]);pos+=o*v
  for mode in (1,2);push!(fs,(idx,src,init,CUDA.fill(-8f0,o*v+17),Vector{Float32}(undef,o*v+17),Int32(n),Int32(o),Int32(v),mode));end
 end
 @assert pos==length(ST_INPUT)+1;fs
end
function reset!(f)
 CUDA.fill!(f[4],-8f0);copyto!(f[4],1,f[3],1,length(f[3]));CUDA.synchronize()
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);@assert rep==3;fs=prepare();for f in fs;reset!(f);launch!(f);CUDA.synchronize();end;got=Vector{Float32}(undef,length(ST_GOLD))
 for trial in 1:5
  times=Float64[];offset=1;gct=0.;gcp=0
  for f in fs
   reset!(f);gb=Base.gc_num();t=time_ns();for _ in 1:rep;launch!(f);end;CUDA.synchronize();push!(times,(time_ns()-t)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   copyto!(f[5],f[4]);copyto!(got,offset,f[5],1,length(f[5]));offset+=length(f[5])
  end
  @assert offset==length(got)+1;ok=true;ma=0.
  for i in eachindex(got);e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);ok&=isfinite(got[i])&&e==0.;end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0,\"timing_region\":\"sum10_shared_atomic_scatter_launchmeans\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
