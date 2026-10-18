using CUDA,Printf
function arithmetic_study(launch)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));pos=2;cs=Any[]
 for ci in 1:Int(prefix[1])
  bits,N,op,L,R=Int.(prefix[pos:pos+4]);v1=prefix[pos+5];v2=prefix[pos+6];pos+=7;T=bits==32 ? Float32 : Float64;h=vcat(fill(T(-12345),8),T.(prefix[pos:pos+N-1]),fill(T(-12345),8));pos+=N;d=CuArray(h);push!(cs,(bits=bits,N=N,op=op,L=L,R=R,v1=T(v1),v2=T(v2),h=h,d=d,data=view(d,9:N+8)))
 end
 @assert pos==length(prefix)+1
 for c in cs;launch(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);ok=true;go=0;times=Float64[];ma=0.;mr=0.
  for c in cs
   copyto!(c.d,c.h);CUDA.synchronize();before=Array(c.d);ok&=reinterpret(UInt8,before)==reinterpret(UInt8,c.h);t=time_ns();launch(c);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   got=Float64.(Array(c.d));ref=gold[go+1:go+c.N+16];go+=c.N+16;x=got[9:c.N+8];r=ref[9:c.N+8];er=abs.(x.-r);at,rt=c.bits==32 ? (1e-4,1e-5) : (1e-11,1e-12);exact=c.R==0||c.op in (1,2)
   ok&=all(isfinite,got)&&got[1:8]==ref[1:8]&&got[c.N+9:end]==ref[c.N+9:end]&&(exact ? x==r : all(er .<= at .+rt.*abs.(r)));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(r).+1e-300)));append!(art,got)
   ref=gold[go+1:go+c.N+16];go+=c.N+16;got=Float64.(before);ok&=got==ref;append!(art,got)
  end
  @assert go==length(gold);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum192_full_source_arithmetic_network_launches\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",go,ma,mr,file,length(art),string(times))
 end
end
