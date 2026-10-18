using CUDA,Printf
@inline function binary_log_study(input::Float32,p::Int32)
 exponent=Int32((reinterpret(UInt32,input)&UInt32(0x7f800000))>>23)-Int32(127)
 y=input/Float32(Int32(1)<<exponent);s=Int32(0);result=0f0
 while s<p+Int32(1)&&y!=1f0
  m=Int32(0)
  while y<2f0&&s+m<p+Int32(1);y*=y;m+=Int32(1);end
  s+=m;denom=UInt64(1)<<s
  if s>p;break;end
  result+=1f0/Float32(denom);y/=2f0
 end
 Float32(exponent)+result
end
function log_study(factory)
 host=collect(reinterpret(Float32,read(joinpath(@__DIR__,"study_input_f32.bin"))))
 oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))))
 ps=Int32[vcat(collect(1:16),[18,20,23])...];n=length(host);np=length(ps);repeats=17;guard=-12345.5f0
 input=CuArray(host);out=CUDA.fill(guard,n*np+2);launch! =factory(input,out,Int32(n))
 function suite!()
  for (row,p) in enumerate(ps),_ in 1:repeats;launch!(Int32(row-1),p);end
  CUDA.device_synchronize(blocking=true)
 end
 suite!();allok=true
 for trial in 1:5
  fill!(out,guard);CUDA.device_synchronize(blocking=true);t=time_ns();suite!();secs=(time_ns()-t)*1e-9/repeats
  got=Array(out);back=Array(input);ok=back==host&&got[1]==guard&&got[end]==guard;maxerr=0.0;bad=0
  for (row,p) in enumerate(ps),i in 1:n
   nearest=Float32(oracle[i]);bound=2.0^(-p)+16*2.0^-24+2*Float64(nextfloat(nearest)-nearest)
   err=abs(Float64(got[1+(row-1)*n+i])-oracle[i]);maxerr=max(maxerr,err)
   if !isfinite(got[1+(row-1)*n+i])||err>bound;bad+=1;end
  end
  ok &=bad==0;allok &=ok;artifact=vcat(back,Float32.(ps),got)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"independent_accuracy_failures\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"timing_region\":\"19precision_log_approx_kernels_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(artifact),bad,maxerr,repeats,path,length(artifact))
 end
 allok||error("independentpointwiseaccuracy/input/guardcontractfailed");println("PASS")
end
