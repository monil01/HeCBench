using CUDA,Printf
function kurtosis_study(launch!)
 @assert sizeof(KStore)==24&&sizeof(KResult)==20
 records=collect(reinterpret(KStore,read(joinpath(@__DIR__,"fixtures_aos24.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));sizes=[1,257,4099,4099,513,513,1024];G=16;R=10;guard=KStore(Int32(-777),Int32(19),UInt64(65261),-1234.5f0,UInt32(0));allok=true
 function operation(x,n,repeats)
  B=cld(n,256);last=zero(KResult)
  for r in 1:repeats
   partial=CuArray{KResult}(undef,B);out=CuArray{KResult}(undef,1);launch!(x,partial,out,n,B);last=Array(out)[1];CUDA.unsafe_free!(partial);CUDA.unsafe_free!(out)
  end
  Ref(last)
 end
 for trial in 0:5
  prefix=Float64[];outputs=Float64[];times=Float64[];ok=true;offset=0;maxerr=0.
  for (ci,n) in enumerate(sizes)
   host=vcat(fill(guard,G),records[offset+1:offset+n],fill(guard,G));dev=CuArray(host);x=view(dev,G+1:G+n);values=Float64.(getfield.(records[offset+1:offset+n],:value));mean=sum(values)/n;delta=1e-5+4e-6*max(1,maximum(abs,values));sums=[sum(abs(v-mean)^power for v in values) for power in 1:4];budgets=vcat(0.,delta,[3e-5*(1+sums[power])+power*delta*sums[power-1] for power in 2:4]);ref=oracle[5ci-4:5ci]
   if trial==0;operation(x,n,1);CUDA.device_synchronize(blocking=true);offset+=n;continue;end
   CUDA.device_synchronize(blocking=true);start=time_ns();result=operation(x,n,R);CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/R);a=result[];out=Float64[a.count,a.mean,a.m2,a.m3,a.m4];actual=Array(dev);bits=collect(reinterpret(UInt32,actual));ok &=bits==reinterpret(UInt32,host)&&a.count==n
   for field in 2:5;e=abs(out[field]-ref[field]);maxerr=max(maxerr,e);ok &=isfinite(out[field])&&e<=budgets[field];end
   if ci in [1,5,6];ok &=a.m2==0&&a.m3==0&&a.m4==0;end
   append!(prefix,Float64.(bits));append!(outputs,vcat(Float64(sizeof(KResult)),out));offset+=n
  end
  trial==0&&continue
  allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum7_full_reduce_API_hostwindows_R10\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),maxerr,join(times,","),path,length(artifact))
 end
 allok||error("Centralmoments/independentoracle/input/guardcheck failed");println("PASS")
end
