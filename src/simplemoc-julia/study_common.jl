using CUDA,Printf
include("attenuation_device.jl")
function guarded(::Type{T},values) where T
 host=vcat(fill(T(16384),8),T.(values),fill(T(16384),8));base=CuArray(host)
 (host=host,base=base,data=view(base,9:length(host)-8),n=length(values))
end
function simplemoc_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==6023&&pre[1]==6&&length(gold)==13164
 pos=2;cases=Any[]
 for ci in 1:6
  R,F,G,Q,FAI,repeat,mode=Int.(pre[pos:pos+6]);pos+=7
  ns=[R*F*G,R*F*G,R*G,G];a=Any[]
  for n in ns;push!(a,guarded(Float32,pre[pos:pos+n-1]));pos+=n;end
  src,flux,sig,state=a;scratch=guarded(Float32,fill(-777f0,14*G));qid=guarded(Int32,[Q]);fid=guarded(Int32,[FAI])
  push!(cases,(R=R,F=Int32(F),G=Int32(G),Q=Q,FAI=FAI,repeat=repeat,mode=mode,src=src,flux=flux,sig=sig,state=state,scratch=scratch,qid=qid,fid=fid))
 end
 @assert pos==length(pre)+1
 function execute(c)
  for a in (c.flux,c.state,c.scratch);copyto!(a.base,a.host);end
  beforef=Array(c.flux.base);befores=Array(c.state.base);ok=beforef==c.flux.host&&befores==c.state.host;CUDA.synchronize()
  start=time_ns();for rep in 1:c.repeat;launch(c);end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9
  arrays=(c.src,c.flux,c.sig,c.state,c.scratch,c.qid,c.fid);src,flux,sig,state,scratch,qid,fid=map(a->Array(a.base),arrays)
  ok &=src==c.src.host&&sig==c.sig.host&&qid==c.qid.host&&fid==c.fid.host
  for r in 0:c.R-1,j in 0:Int(c.F)-1
   if r!=c.Q||j!=c.FAI
    first=(r*c.F+j)*c.G+9;last=first+c.G-1;ok &=flux[first:last]==c.flux.host[first:last]
   end
  end
  guards=Float64[]
  for a in (src,flux,sig,state,scratch,qid,fid);g=vcat(a[1:8],a[end-7:end]);ok &=all(==(16384),g);append!(guards,g);end
  vals=vcat(Float64.(flux[9:end-8]),Float64.(state[9:end-8]),Float64.(scratch[9:end-8]),Float64.(src[9:end-8]),Float64.(sig[9:end-8]),Float64.(beforef[9:end-8]),Float64.(befores[9:end-8]),Float64.(qid[9:end-8]),Float64.(fid[9:end-8]),guards)
  ok &=all(x->x>=-2e-5,state[9:end-8])&&all(x->x>=-2e-5,scratch[9+8*c.G:8+9*c.G])
  (vals=vals,seconds=seconds,ok=ok)
 end
 for c in cases;@assert execute(c).ok;end
 for trial in 1:5
  art=copy(pre);times=Float64[];ok=true
  for c in cases;z=execute(c);append!(art,z.vals);push!(times,z.seconds);ok &=z.ok;end
  actual=art[length(pre)+1:end];errors=abs.(actual.-gold);ok &=all(isfinite,actual)&&all(errors.<=2e-5.+2e-5.*abs.(gold));ma=maximum(errors)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum6_corrected_single_segment_attenuation_cases_repeat_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"case_internal_repeats\":[1,2,3,2,3,2],\"blocks\":128,\"threads\":128,\"active_segments_per_case\":1}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,path,length(art),string(times))
 end
end
