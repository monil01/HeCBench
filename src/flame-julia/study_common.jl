using CUDA,Printf
const ROOT=@__DIR__
const POINT_GUARD=(Float16(-777),Float16(-777))
const VERTEX_GUARD=(-777f0,-777f0,-777f0)
function params_probe_kernel!(out,p)
 params_to_words!(out,p);return
end
function flatten_values(a)
 Float64[x for item in a for x in item]
end
function validate_output(got,gold)
 exact=0;ma=0.;mr=0.;worst=0.;offset=0;ok=all(isfinite,got)
 for case in 0:1
  for (kind,count,components,guard) in [(:points,256,2,8),(:colors,256,1,8),(:vertices,16384,3,8)]
   size=(count+2guard)*components
   for j in 0:size-1
    a=got[offset+j+1];b=gold[offset+j+1]
    isguard=j<guard*components || j>=(guard+count)*components
    iscolor=kind==:colors || (kind==:vertices && j%3==2)
    if isguard || iscolor || case==1
     exact+=a!=b;ok &= a==b
    else
     er=abs(a-b);ma=max(ma,er);mr=max(mr,er/max(abs(b),1e-30));frac=er/(2.0^-9+2.0^-9*abs(b));worst=max(worst,frac);ok &= frac<=1
    end
   end
   offset+=size
  end
 end
 @assert offset==length(gold)
 ok,ma,mr,worst,exact
end
function study_main(model)
 fixture=collect(reinterpret(Float64,read(joinpath(ROOT,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(ROOT,"oracle.bin"))))
 params=[only(reinterpret(FlameParams,read(joinpath(ROOT,name)))) for name in ["params_default.bin","params_linear.bin"]]
 for p in params
  @assert p.mapping[end]==256 && issorted(collect(p.mapping))
  @assert all(v->abs(v.factor)<.01f0 || v.idx in (0,5,7,9,13),p.variations)
 end
 start=collect(reinterpret(NTuple{2,Float32},read(joinpath(ROOT,"start.bin"))));randoms=collect(reinterpret(Float32,read(joinpath(ROOT,"random.bin"))));perms=collect(reinterpret(UInt16,read(joinpath(ROOT,"permutations.bin"))))
 @assert all(sort(perms[i*256+1:(i+1)*256])==UInt16.(0:255) for i in 0:127)
 sw=CuArray(vcat(fill((-777f0,-777f0),8),start,fill((-777f0,-777f0),8)));rw=CuArray(vcat(fill(-777f0,8),randoms,fill(-777f0,8)));tw=CuArray(vcat(fill(UInt16(0xa5a5),8),perms,fill(UInt16(0xa5a5),8)))
 s=@view sw[9:end-8];r=@view rw[9:end-8];t=@view tw[9:end-8]
 pw=CUDA.fill(POINT_GUARD,272);cw=CUDA.fill(Float16(-777),272);vw=CUDA.fill(VERTEX_GUARD,16400)
 points=@view pw[9:end-8];colors=@view cw[9:end-8];vertices=@view vw[9:end-8];probe=CuArray{UInt32}(undef,482)
 function frame(p)
  launch_initialize(model,points,colors,t,Int32(0),s,r,p);CUDA.synchronize()
  for step in Int32(1):Int32(15);launch_iterate(model,points,colors,t,step,r,p);CUDA.synchronize();end
  launch_generate(model,vertices,points,colors,t,Int32(16),r,p);CUDA.synchronize()
 end
 for p in params
  @cuda threads=1 blocks=1 params_probe_kernel!(probe,p)
  frame(p)
 end
 CUDA.synchronize()
 expected_echo=vcat(fixture[7:970],fill(-777.,16),fixture[971:1482],fill(-777.,16),fill(-777.,8),fixture[1483:2506],fill(-777.,8),fill(42405.,8),fixture[2507:end],fill(42405.,8))
 for trial in 1:5
  words=Float64[]
  for p in params
   @cuda threads=1 blocks=1 params_probe_kernel!(probe,p)
   append!(words,Float64.(Array(probe)))
  end
  prefix=vcat(Float64[256,64,15,1024,128,2],words,flatten_values(Array(s)),Float64.(Array(r)),Float64.(Array(t)))
  actual_echo=vcat(words,flatten_values(Array(sw)),Float64.(Array(rw)),Float64.(Array(tw)))
  got=Float64[];elapsed=0.;times=Float64[]
  for p in params
   fill!(pw,POINT_GUARD);fill!(cw,Float16(-777));fill!(vw,VERTEX_GUARD);CUDA.synchronize()
   stamp=time_ns();frame(p);secs=(time_ns()-stamp)*1e-9;elapsed+=secs;push!(times,secs)
   append!(got,flatten_values(Array(pw)));append!(got,Float64.(Array(cw)));append!(got,flatten_values(Array(vw)))
  end
  ok,ma,mr,frac,exact=validate_output(got,gold)
  ok &= reinterpret(UInt64,prefix)==reinterpret(UInt64,fixture) && reinterpret(UInt64,actual_echo)==reinterpret(UInt64,expected_echo)
  output=vcat(prefix,got,actual_echo);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,output)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"max_engineering_budget_fraction\":%.17g,\"exact_color_guard_linear_mismatches\":%d,\"path_times_seconds\":[%s],\"timing_region\":\"sum2_full_frame_17kernel_hostinclusive_regions\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(output),ma,mr,frac,exact,join(string.(times),","),file,length(output))
 end
end
