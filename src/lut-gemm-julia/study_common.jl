using CUDA,Printf
function lut_study(launch)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));pos=2;cs=Any[]
 for ci in 1:Int(prefix[1])
  M,K,nb,G,bi,T=Int.(prefix[pos:pos+5]);pos+=6;ns=[K÷32*nb*M,K÷G*nb*M,bi==1 ? K÷G*M : 0,K];hs=Any[];ds=Any[];vs=Any[]
  for (i,n) in enumerate(ns)
   ty=i==1 ? UInt32 : Float16;guard=ty(i==1 ? 1515870810 : -123);h=vcat(fill(guard,8),ty.(prefix[pos:pos+n-1]),fill(guard,8));pos+=n;d=CuArray(h);push!(hs,h);push!(ds,d);push!(vs,view(d,9:8+n))
  end
  yo=CUDA.fill(Float16(-123),M+16);y=view(yo,9:M+8);push!(cs,(M=M,K=K,nb=nb,G=G,bi=bi,T=T,hs=hs,ds=ds,vs=vs,yo=yo,y=y))
 end
 @assert pos==length(prefix)+1
 for c in cs;fill!(c.y,Float16(0));launch(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);go=0;ok=true;times=Float64[]
  for c in cs
   fill!(c.y,Float16(.5));CUDA.synchronize();t=time_ns();fill!(c.y,Float16(0));launch(c);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   got=Float64.(Array(c.yo));ref=gold[go+1:go+length(got)];go+=length(got);ok&=all(isfinite,got)&&got==ref;append!(art,got)
   for (h,d) in zip(c.hs,c.ds)
    a=Array(d);got=Float64.(a);ref=gold[go+1:go+length(got)];go+=length(got);ok&=reinterpret(UInt8,a)==reinterpret(UInt8,h)&&got==ref;append!(art,got)
   end
  end
  @assert go==length(gold)&&length(art)==length(prefix)+length(gold);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum7_clear_plus_packed_half_LUT_GEMV\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",go,file,length(art),string(times))
 end
end
