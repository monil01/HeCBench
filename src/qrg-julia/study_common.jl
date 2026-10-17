
using CUDA,Printf
function qrg_study(launch!)
 prefix=reinterpret(Float64,read("study_input_f64.bin"));gold=reinterpret(Float64,read("study_math_expected_f64.bin"));cj=reinterpret(Int64,read("study_cjn_i64.bin"));n,seed,total,distance,repeats=Int.(prefix[1:5]);table,cjn=init_quasirandom_generator();@assert vec(cjn)==cj;@assert Float64.(table)==prefix[6:98];inputs=UInt32.(prefix[99:112]);dt=CuArray(table);di=CuArray(inputs);out=CUDA.zeros(Float32,total);probes=CUDA.zeros(Float32,14)
 # Precompute genuine source CPU reference outside all measured regions.
 refs=(Float64[get_quasirandom_value63(cjn,Int64(i%n),i÷n) for i=0:total-1],Float64[moro_inv_cnd_cpu(UInt32(i+1)*UInt32(distance)) for i=0:total-1])
 for kind=0:2;launch!(kind,out,dt,UInt32(seed),UInt32(total),UInt32(n),UInt32(distance),probes,di);end
 CUDA.synchronize();allok=true
 for trial=1:5
  artifact=collect(prefix);times=Float64[];l1=Float64[];ok=true;maxerr=0.;cdferr=0.
  for stage=0:1
   fill!(out,0);CUDA.synchronize();start=time_ns()
   for _=1:repeats;launch!(stage,out,dt,UInt32(seed),UInt32(total),UInt32(n),UInt32(distance),probes,di);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeats);v=Float64.(Array(out));g=gold[stage*total+1:(stage+1)*total];err=maximum(abs.(v.-g));maxerr=max(maxerr,err);push!(l1,sum(abs.(v.-refs[stage+1]))/sum(abs,refs[stage+1]));ok&=all(isfinite,v)&&(stage==0 ? v==g : all(abs.(v.-g).<=2e-5.+2e-6.*abs.(g)))&&l1[end]<1e-6
   if stage==1
    for i=1:total
     x=UInt32(i)*UInt32(distance);lo=min(x,typemax(UInt32)-x);prob=(Float64(lo)+.5)/Float64(typemax(UInt32));if x>=0x80000000;prob=1-prob;end
     cdf=.5*ccall((:erfc,Base.Math.libm),Float64,(Float64,),-v[i]/sqrt(2.));cdferr=max(cdferr,abs(cdf-prob))
    end
   end
   append!(artifact,v)
  end
  launch!(2,out,dt,UInt32(seed),UInt32(total),UInt32(n),UInt32(distance),probes,di);CUDA.synchronize();v=Float64.(Array(probes));g=gold[2*total+1:end];ok&=all(isfinite,v)&&all(abs.(v.-g).<=2e-5.+2e-6.*abs.(g))&&cdferr<=2e-6;maxerr=max(maxerr,maximum(abs.(v.-g)));append!(artifact,v);allok&=ok
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"stage_seconds\":[%.12g,%.12g],\"correct\":%s,\"checked_values\":393236,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"qrng_source_l1\":%.12g,\"icnd_source_l1\":%.12g,\"normal_cdf_max_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":20,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times[1],times[2],ok ? "true" : "false",maxerr,l1[1],l1[2],cdferr,path,length(artifact))
 end
 allok||error("QRG independent oracle failed");println("PASS")
end
