using CUDA,Printf
function gpp_study(launch!)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==8955&&length(gold)==9854
 pos=2;cs=Any[]
 for ci in 1:Int(prefix[1])
  nb,ng,nc,gx,gy=Int32.(prefix[pos:pos+4]);pos+=5;lens=[ng,nc+1,3,ng*nc,ng*nc,nb*nc,nb*nc,nb*nc,nb*nc,ng*nc,ng*nc,nc];hs=Any[];ds=Any[]
  for (k,n) in enumerate(lens)
   T=k<=2 ? Int32 : Float64;h=vcat(fill(T(-12345),8),T.(prefix[pos:pos+n-1]),fill(T(-12345),8));pos+=n;push!(hs,h);push!(ds,CuArray(h))
  end
  outs=[CUDA.fill(-12345.,19),CUDA.fill(-12345.,19)];args=(nb,ng,nc,(view(x,9:length(x)-8) for x in ds)...,(view(x,9:11) for x in outs)...);push!(cs,(nb=nb,ng=ng,nc=nc,gx=gx,gy=gy,hs=hs,ds=ds,outs=outs,args=args))
 end
 @assert pos==8955+1
 reset(c)=foreach(x->fill!(view(x,9:11),0.),c.outs)
 for c in cs;reset(c);launch!(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);elapsed=0.;paths=Float64[];ok=true;ma=0.;mr=0.;go=0
  for c in cs
   reset(c);CUDA.synchronize();t=time_ns();launch!(c);CUDA.synchronize();dt=(time_ns()-t)*1e-9;elapsed+=dt;push!(paths,dt)
   for x in c.outs
    got=Array(x);ref=gold[go+1:go+19];go+=19;ok&=all(isfinite,got)&&got[1:8]==ref[1:8]&&got[12:19]==ref[12:19]&&all(abs.(got[9:11].-ref[9:11]).<=1e-11 .+1e-11.*abs.(ref[9:11]));ma=max(ma,maximum(abs.(got[9:11].-ref[9:11])));mr=max(mr,maximum(abs.(got[9:11].-ref[9:11])./(abs.(ref[9:11]).+1e-300)));append!(art,got)
   end
   for (h,d) in zip(c.hs,c.ds)
    got=Array(d);ref=gold[go+1:go+length(got)];go+=length(got);ok&=reinterpret(UInt8,got)==reinterpret(UInt8,h)&&Float64.(got)==ref;append!(art,Float64.(got))
   end
  end
  @assert go==9854&&length(art)==18809;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":9854,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum4_single_gpp_complex_reduction_kernels\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":18809,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,elapsed,ok ? "true" : "false",ma,mr,file,string(paths))
 end
end
