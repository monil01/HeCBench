using CUDA,Printf
include(joinpath(@__DIR__,"metadata_math.jl"))
const INPUT=collect(reinterpret(Int32,read(joinpath(@__DIR__,"fixture_bits.bin"))))
const GOLD=collect(reinterpret(Int32,read(joinpath(@__DIR__,"expected_metadata.bin"))))
const GUARD=Int32(-777777)
function fixtures()
 fs=[];pos=1
 for _ in 1:2
  B,H,R,M,BN,V,S,C=Int64.(INPUT[pos:pos+7]);pos+=8
  vals=[]
  for len in (B,B,B*H*V,B*H*S,H,H)
   push!(vals,INPUT[pos:pos+len-1]);pos+=len
  end
  push!(fs,(B,H,R,M,BN,V,S,C,vals))
 end
 @assert pos==length(INPUT)+1
 fs
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fs=fixtures();paths=[]
 for f in fs
  ins=[CuArray(a) for a in f[9]];scalar=f[1]*f[2]*f[3]
  for causal in (true,false)
   lens=(scalar,scalar*f[7],scalar,scalar*f[6]);outs=[CuArray(vcat(zeros(Int32,n),fill(GUARD,17))) for n in lens];push!(paths,(f,ins,outs,causal))
  end
 end
 for p in paths;launch!(p);end;CUDA.synchronize()
 for trial in 1:5
  for p in paths,a in p[3];copyto!(a,vcat(zeros(Int32,length(a)-17),fill(GUARD,17)));end;CUDA.synchronize();times=Float64[];gct=0.;gcp=0
  for p in paths
   CUDA.synchronize();gb=Base.gc_num();st=time_ns();for _ in 1:rep;launch!(p);end;CUDA.synchronize();push!(times,(time_ns()-st)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
  end
  got=vcat([Array(a) for p in paths for a in p[3]]...);ok=got==GOLD;ma=maximum(abs.(Int64.(got).-Int64.(GOLD)));artifact=vcat(INPUT,got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"timing_region\":\"sum4_vertical_slash_metadata_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,rep,join(times,","),gct,gcp,file,length(artifact))
 end
end
