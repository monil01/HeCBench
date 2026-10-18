using CUDA,Printf
include("study_device.jl")
function winograd_study(factory)
 input=reinterpret(Float32,read(joinpath(@__DIR__,"fixture_f32.bin")));expected=reinterpret(Float64,read(joinpath(@__DIR__,"expected_f64.bin")));n=129^2;cases=[];prefix=Float32[]
 for z in 0:1
  at=z*(16+131^2)+1;filter=collect(input[at:at+15]);image=collect(input[at+16:at+15+131^2]);da=CuArray(image);df=CuArray(filter);dout=CuArray{Float32}(undef,n);host=Vector{Float32}(undef,n);actual=similar(host);ref=similar(host);cpu_tiles!(image,filter,ref,65);launch=[factory(da,df,dout,(p*3÷100)*32)for p in 0:100];append!(prefix,Array(df));append!(prefix,Array(da));push!(cases,(image,filter,dout,host,actual,ref,launch))
 end
 prefix==input||error("inputidentity")
 function co!(c,p)
  image,filter,dout,host,actual,ref,launch=c;cpu=(p*3÷100)*32;f=launch[p+1];start=time_ns()
  if f!==nothing;f();end
  if cpu>0
   cpu_tiles!(image,filter,host,cpu);count=f===nothing ? n : cpu*2*129
   copyto!(dout,1,host,1,count)
  end
  copyto!(actual,dout)
  return (time_ns()-start)*1e-9
 end
 sourcecheck(ref,actual)=all(eachindex(ref))do k
  a=ref[k];b=actual[k];abs(a)<.01f0&&abs(b)<.01f0 || 100f0*abs(abs(a-b)/abs(a+1f-8))<=1.05f0
 end
 for c in cases,p in 0:100;co!(c,p);end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  out=Vector{Float32}(undef,length(prefix)+2*101*n);copyto!(out,1,prefix,1,length(prefix));times=zeros(Float64,2);ok=true;sourceok=true;err=0.;scaled=0.
  for z in 1:2,p in 0:100
   c=cases[z];times[z]+=co!(c,p);actual=c[5];sourceok&=sourcecheck(c[6],actual);copyto!(out,length(prefix)+((z-1)*101+p)*n+1,actual,1,n)
   for k in 1:n
    e=abs(Float64(actual[k])-expected[(z-1)*n+k]);tol=1e-5+1e-5*abs(expected[(z-1)*n+k]);ok&=isfinite(actual[k])&&e<=tol;err=max(err,e);scaled=max(scaled,e/tol)
   end
  end
  ok&=sourceok;allok&=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"winograd_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g],\"correct\":%s,\"source_percent_checker_pass\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"max_scaled_absolute_oracle_error\":%.12g,\"internal_repeats\":1,\"jit_excluded\":true,\"timing_region\":\"sum_two_full101CPU_GPU_split_sweeps_CPUwork_prefixH2D_fullD2H\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),string(sourceok),2*101*n,err,scaled,file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
