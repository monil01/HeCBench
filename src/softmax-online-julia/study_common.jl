using CUDA,Printf
const ST_INPUT=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=collect(reinterpret(Float64,read(joinpath(@__DIR__,"expected_outputs.bin"))))
const ST_WIDTHS=(32,64,128,256,512,1024)
function prepare()
 frames=Any[];pos=2
 for _ in 1:Int(ST_INPUT[1])
  n,c,len=Int.(ST_INPUT[pos:pos+2]);pos+=3;inp=CuArray(ST_INPUT[pos:pos+len-1]);pos+=len
  push!(frames,(inp,CUDA.fill(-12345f0,len),Int32(n),Int32(c),Vector{Float32}(undef,len)))
 end
 @assert pos==length(ST_INPUT)+1
 frames
end
function study_main(model,launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);frames=prepare()
 for f in frames,w in ST_WIDTHS;launch!(f,w);CUDA.synchronize();end
 got=Vector{Float32}(undef,length(ST_GOLD))
 for trial in 1:5
  times=Float64[];offset=1;gct=0.;gcp=0;ok=true;mass=0.;shiftmax=0.
  for f in frames,w in ST_WIDTHS
   CUDA.fill!(f[2],-12345f0);CUDA.synchronize();gb=Base.gc_num();t=time_ns()
   for _ in 1:rep;launch!(f,w);end
   CUDA.synchronize();push!(times,(time_ns()-t)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
   copyto!(f[5],f[2]);host=f[5];n,c=Int(f[3]),Int(f[4]);copyto!(got,offset,host,1,length(host));offset+=length(host)
   for row in 0:n-1
    e=abs(sum(Float64(host[row*c+j]) for j in 1:c)-1);mass=max(mass,e);ok&=e<=1e-5
    for j in 1:c;v=host[row*c+j];ok&=isfinite(v)&&0<=v<=1f0+1f-6;end
   end
   for dst in (2,3)
    if dst<n
     for j in 1:c;ea=abs(Float64(host[c+j])-Float64(host[dst*c+j]));shiftmax=max(shiftmax,ea);ok&=reinterpret(UInt32,host[c+j])==reinterpret(UInt32,host[dst*c+j]);end
    end
   end
  end
  @assert offset==length(got)+1
  ma=0.;mr=0.
  for i in eachindex(got)
   e=abs(Float64(got[i])-ST_GOLD[i]);ma=max(ma,e);mr=max(mr,e/max(abs(ST_GOLD[i]),1e-12));ok&=isfinite(got[i])&&(ST_GOLD[i]==-12345 ? got[i]==-12345 : e<=1e-4+abs(ST_GOLD[i])*Float64(eps(Float32)))
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,got))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"max_row_sum_abs_error\":%.17g,\"max_exact_shift_pair_abs_error\":%.17g,\"timing_region\":\"sum30_source_block_online_kernel4_paths\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,mass,shiftmax,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end
