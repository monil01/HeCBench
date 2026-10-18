using CUDA,Printf
function lookback_study(launch,probe)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==8199&&length(gold)==20096&&prefix[1:3]==[256.,4.,512.]
 hs=vcat(fill(UInt32(0x5a5a5a5a),8),UInt32.(prefix[4:1027]),fill(UInt32(0x5a5a5a5a),8));seedsO=CuArray(hs);seeds=view(seedsO,9:1032);cs=Any[];pos=1028
 for ci in 1:4
  T=UInt32(prefix[pos]);pos+=1;@assert T<=15;hh=Any[];ds=Any[]
  for k in 1:7
   h=vcat(fill(-12345f0,8),Float32.(prefix[pos:pos+255]),fill(-12345f0,8));pos+=256;push!(hh,h);push!(ds,CuArray(h))
  end
  mo=CUDA.fill(-12345f0,272);vo=similar(mo);pro=CUDA.fill(-12345.,1536);ivs=[view(d,9:264) for d in ds];m=view(mo,9:264);v=view(vo,9:264);args=(T,seeds,m,v,ivs...);push!(cs,(T=T,hs=hh,ds=ds,mo=mo,vo=vo,probe=pro,seeds=seeds,args=args))
 end
 @assert pos==8199+1
 for c in cs;launch(c);probe(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);times=Float64[];ok=true;ma=0.;mr=0.;go=0
  for (ci,c) in enumerate(cs)
   fill!(c.mo,-12345f0);fill!(c.vo,-12345f0);CUDA.synchronize();t=time_ns();launch(c);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   for d in [c.mo,c.vo]
    got=Float64.(Array(d));ref=gold[go+1:go+272];go+=272;ok&=all(isfinite,got)&&got[1:8]==ref[1:8]&&got[265:272]==ref[265:272];x=got[9:264];r=ref[9:264]
    if ci>=3;ok&=x==r
    else;er=abs.(x.-r);ok&=all(>=(0),x)&&all(er.<=.002 .+.002.*abs.(r));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(r).+1e-300)));end
    append!(art,got)
   end
   probe(c);CUDA.synchronize();pr=Array(c.probe);ref=gold[go+1:go+1536];go+=1536;ok&=pr==ref;append!(art,pr);ss=Array(seedsO);ref=gold[go+1:go+1040];go+=1040;ok&=ss==hs&&Float64.(ss)==ref;append!(art,Float64.(ss))
   for (h,d) in zip(c.hs,c.ds)
    x=Array(d);ref=gold[go+1:go+272];go+=272;ok&=reinterpret(UInt8,x)==reinterpret(UInt8,h)&&Float64.(x)==ref;append!(art,Float64.(x))
   end
  end
  @assert go==20096&&length(art)==28295;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":20096,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum4_full512path_lookbacklike_simulation_kernels\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":28295,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"paths_per_parameter\":512,\"RNG_probe_outside_timing\":true}\n",trial,sum(times),ok ? "true" : "false",ma,mr,file,string(times))
 end
end
