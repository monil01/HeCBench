using CUDA,Printf
function doh_study(launch!)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==2722&&length(gold)==11352
 cases=Any[];off=0
 for ci in 0:3
  h=Int32(prefix[2711+2ci]);w=Int32(prefix[2712+2ci]);n=Int(h*w);host=vcat(fill(-12345f0,8),Float32.(prefix[off+n+1:off+2n]),fill(-12345f0,8));off+=2n
  input=CuArray(host);output=CUDA.fill(-12345f0,n+16);push!(cases,(h=h,w=w,n=n,host=host,input=input,a=view(input,9:n+8),output=output,o=view(output,9:n+8)))
 end
 @assert off==2710
 sigmas=Float32.(prefix[2719:2722])
 for c in cases,sigma in sigmas;launch!(c.a,c.h,c.w,sigma,c.o,c.n);CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);elapsed=0.;path_times=Float64[];ok=true;err=0.;rel=0.;go=0
  for c in cases,sigma in sigmas
   fill!(c.output,-12345f0);CUDA.synchronize();t=time_ns()
   for rep in 1:100;launch!(c.a,c.h,c.w,sigma,c.o,c.n);end
   CUDA.synchronize();dt=(time_ns()-t)*1e-9/100;elapsed+=dt;push!(path_times,dt)
   got=Float64.(Array(c.output));ref=gold[go+1:go+c.n+16];go+=c.n+16
   ok&=all(isfinite,got)&&got[1:8]==ref[1:8]&&got[end-7:end]==ref[end-7:end]&&all(abs.(got.-ref).<=1e-4 .+2e-6.*abs.(ref));err=max(err,maximum(abs.(got.-ref)));rel=max(rel,maximum(abs.(got.-ref)./(abs.(ref).+1e-300)));append!(art,got)
   echo=Array(c.input);ok&=reinterpret(UInt32,echo)==reinterpret(UInt32,c.host);ref=gold[go+1:go+c.n+16];go+=c.n+16;ok&=Float64.(echo)==ref;append!(art,Float64.(echo))
  end
  @assert go==11352&&length(art)==14074;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":11352,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum16_box_hessian_kernel_batches_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":14074,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,elapsed,ok ? "true" : "false",err,rel,file,string(path_times))
 end
end
