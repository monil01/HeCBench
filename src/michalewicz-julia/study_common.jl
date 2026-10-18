using CUDA,Printf
@inline function mx_value(values,start::Int32,dim::Int32)
 result=0f0
 for j in Int32(1):dim
  @inbounds x=values[start+j]
  a=sin(x);b=sin((Float32(j)*x*x)/Float32(pi));result+=a*(b^20f0)
 end
 -result
end
function michalewicz_study(factory)
 input=collect(reinterpret(Float32,read(joinpath(@__DIR__,"input_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));G=4;guard=-1234567f0;gb=reinterpret(UInt32,guard);R=100;allok=true
 for trial in 1:5
  inputoff=0;oracleoff=0;prefix=Float32[];outputs=Float32[];times=Float64[];ok=true;bad=0;maxerr=0.
  for n in [257,4099],dim in [2,5,10],fixture in 0:1
   size=n*dim;host=vcat(fill(guard,G),input[inputoff+1:inputoff+size],fill(guard,G));initial=vcat(fill(gb,G),UInt32(0),fill(gb,G));values=CuArray(host);minbits=CuArray(initial);out=CUDA.fill(guard,n+8);reduce!,capture! =factory(values,minbits,out,Int32(n),Int32(dim))
   reduce!();capture!();CUDA.device_synchronize(blocking=true);copyto!(minbits,initial);CUDA.device_synchronize(blocking=true);actualinitial=Array(minbits);ok &=actualinitial==initial;start=time_ns()
   for _ in 1:R;reduce!();end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/R);capture!();CUDA.device_synchronize(blocking=true);actual=Array(values);ok &=reinterpret(UInt32,actual)==reinterpret(UInt32,host);got=Array(out);mb=Array(minbits);minback=collect(reinterpret(Float32,mb));capturedmin=0f0;expectedmin=0.
   for i in 1:G;bad+=got[i]!=guard||got[n+G+i]!=guard||mb[i]!=gb||mb[G+1+i]!=gb;end
   for i in 1:n
    value=got[G+i];expected=oracle[oracleoff+i];error=abs(Float64(value)-expected);maxerr=max(maxerr,error);bad+=!isfinite(value)||error>3e-5*dim+1e-5*abs(expected)
    if value<capturedmin;capturedmin=value;end
    if expected<expectedmin;expectedmin=expected;end
   end
   bad+=reinterpret(UInt32,capturedmin)!=mb[G+1];error=abs(Float64(minback[G+1])-expectedmin);maxerr=max(maxerr,error);bad+=!isfinite(minback[G+1])||error>3e-5*dim+1e-5*abs(expectedmin)
   if fixture==1;bad+=mb[G+1]!=UInt32(0);end
   append!(prefix,actual);append!(prefix,collect(reinterpret(Float32,actualinitial)));append!(outputs,got);append!(outputs,minback);inputoff+=size;oracleoff+=n
  end
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"timing_region\":\"sum12_source_seeded_GPUmin_windows_per_R100\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),maxerr,join(times,","),path,length(artifact))
 end
 allok||error("objective/seeded GPUmin/oracle/input/guard check failed");println("PASS")
end
