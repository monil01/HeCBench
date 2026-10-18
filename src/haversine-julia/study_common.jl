using CUDA,Printf
@inline function hv_value(a::Float64,b::Float64,c::Float64,d::Float64)
 # Native macro expands lat*M_PI/180.0; retain both sourceoperations.
 ay=(a*3.141592653589793)/180.0;ax=(b*3.141592653589793)/180.0
 by=(c*3.141592653589793)/180.0;bx=(d*3.141592653589793)/180.0
 x=(bx-ax)/2.0;y=(by-ay)/2.0
 sy=sin(y)*sin(y);sx=sin(x)*sin(x);scale=cos(ay)*cos(by)
 2.0*6371.0*asin(sqrt(sy+sx*scale))
end
function haversine_study(factory)
 stress=!isempty(ARGS)&&ARGS[1]=="1";key=stress ? "stress" : "normal";atol=stress ? 1e-3 : 1e-6
 host=collect(reinterpret(Float64,read(joinpath(@__DIR__,"input_$(key)_f64.bin"))))
 expected=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_$(key)_f64.bin"))));n=length(expected);guard=-1234567.25
 inputs=[CuArray(vcat(guard,host[k:4:end],guard)) for k in 1:4];out=CUDA.fill(guard,n+2);repeats=100;launch! =factory(inputs,out,Int32(n))
 function suite!()
  for _ in 1:repeats;launch!();end
  CUDA.device_synchronize(blocking=true)
 end
 suite!();allok=true
 for trial in 1:5
  fill!(out,guard);CUDA.device_synchronize(blocking=true);t=time_ns();suite!();secs=(time_ns()-t)*1e-9/repeats
  actual=[Array(x) for x in inputs];prefix=vec(permutedims(hcat(actual...)))
  targetprefix=vcat(fill(guard,4),host,fill(guard,4));ok=reinterpret(UInt64,prefix)==reinterpret(UInt64,targetprefix)
  got=Array(out);ok &=got[1]==guard&&got[end]==guard;bad=0;nf=0;maxerr=0.0
  for i in 1:n
   if !isfinite(got[i+1]);bad+=1;nf+=1;else;e=abs(got[i+1]-expected[i]);maxerr=max(maxerr,e);bad+=e>atol;end
  end
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"independent_accuracy_failures\":%d,\"nonfinite_outputs\":%d,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"haversine_compute_launches_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok ? "true" : "false",length(artifact),bad,nf,nf==0 ? string(maxerr) : "null",repeats,path,length(artifact))
 end
 allok||error("geometricoracle/input/guardcheck failed");println("PASS")
end
