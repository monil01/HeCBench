using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
function prepare()
 fs=Any[];pos=2
 for _ in 1:Int(ST_INPUT[1])
  batch,dim,L=Int.(ST_INPUT[pos:pos+2]);pos+=3;arr=Any[]
  for j in 1:9;n=Int(ST_INPUT[pos]);pos+=1;push!(arr,CuArray(ST_INPUT[pos:pos+n-1]));pos+=n;end
  for chain in (1,3)
   st=CUDA.fill(-12345f0,length(arr[9])+17);ys=[CUDA.fill(-12345f0,length(arr[1])+17) for _ in 1:chain];hosts=[Vector{Float32}(undef,length(x)) for x in ys];hs=Vector{Float32}(undef,length(st))
   push!(fs,(arr,st,ys,hosts,hs,Int32(batch),Int32(dim),Int32(L),chain))
  end
 end
 @assert pos==length(ST_INPUT)+1;fs
end
function reset!(f)
 CUDA.fill!(f[2],-12345f0);copyto!(f[2],1,f[1][9],1,length(f[1][9]));for y in f[3];CUDA.fill!(y,-12345f0);end;CUDA.synchronize()
end
function study_main(model,launch!)
 fs=prepare();for f in fs;reset!(f);for y in f[3];launch!(f,y);end;CUDA.synchronize();end
 got=Vector{Float32}(undef,length(ST_GOLD))
 for trial in 1:5
  times=Float64[];offset=1;gct=0.;gcp=0
  for f in fs
   reset!(f);gb=Base.gc_num();t=time_ns();for y in f[3];launch!(f,y);end;CUDA.synchronize();push!(times,(time_ns()-t)*1e-9);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   for (y,h) in zip(f[3],f[4]);copyto!(h,y);copyto!(got,offset,h,1,length(h));offset+=length(h);end
   copyto!(f[5],f[2]);copyto!(got,offset,f[5],1,length(f[5]));offset+=length(f[5])
  end
  @assert offset==length(got)+1
  ok=true;ma=0.;mr=0.
  for i in eachindex(got);e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);mr=max(mr,e/max(abs(ST_GOLD[i]),1e-12));ok&=isfinite(got[i])&&(ST_GOLD[i]==-12345 ? got[i]==-12345 : e<=1e-3);end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum6_serialbase_recurrence_chain_paths\",\"jit_excluded\":true,\"internal_repeats\":1,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
