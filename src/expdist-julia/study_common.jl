using CUDA,Printf
function exp_study(factory)
 inputs=collect(reinterpret(Float64,read(joinpath(@__DIR__,"inputs_f64.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));G=16;R=10;allok=true
 for trial in 0:5
  prefix=Float64[];outputs=Float64[];times=Float64[];ok=true;maxerr=0.;maxfraction=0.;io=0;go=0
  for T in (Float32,Float64),(m,n) in ((128,32),(256,64),(256,256)),fixture in 0:2
   nb=(m÷128)*(n÷32);guard=T(-777.25);sentinel=T(-999.5);bufs=Any[];originals=Vector{T}[]
   for count in (2m,2n,m,n)
    h=vcat(fill(guard,G),T.(inputs[io+1:io+count]),fill(guard,G));io+=count;push!(originals,h);push!(bufs,CuArray(h))
   end
   a,b,sa,sb=bufs;ph=vcat(fill(guard,G),fill(sentinel,nb),fill(guard,G));oh=vcat(fill(guard,G),fill(sentinel,1),fill(guard,G));p=CuArray(ph);o=CuArray(oh);launch! =factory(view(a,G+1:G+2m),view(b,G+1:G+2n),view(sa,G+1:G+m),view(sb,G+1:G+n),view(p,G+1:G+nb),view(o,G+1:G+1),m,n)
   if trial==0;launch!();CUDA.device_synchronize(blocking=true);go+=nb+1;continue;end
   beforep=Array(p);beforeo=Array(o);ok &=beforep==ph&&beforeo==oh;CUDA.device_synchronize(blocking=true);start=time_ns();for rep in 1:R;launch!();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/R)
   for (buf,h) in zip(bufs,originals);actual=Array(buf);ok &=actual==h;append!(prefix,Float64.(actual));end
   append!(prefix,Float64.(beforep));append!(prefix,Float64.(beforeo));ap=Array(p);ao=Array(o);ok &=all(==(guard),ap[1:G])&&all(==(guard),ap[end-G+1:end])&&all(==(guard),ao[1:G])&&all(==(guard),ao[end-G+1:end]);vals=vcat(Float64.(ap[G+1:G+nb]),Float64(ao[G+1]));ref=gold[go+1:go+nb+1];go+=nb+1;budget=(T==Float32 ? .001 : 1e-9).+(T==Float32 ? 1e-5 : 1e-12).*abs.(ref);delta=abs.(vals.-ref);ok &=all(isfinite,vals)&&all(delta.<=budget);maxerr=max(maxerr,maximum(delta));maxfraction=max(maxfraction,maximum(delta./budget))
   if fixture==2;ok &=all(==(4096.),vals[1:nb])&&vals[end]==m*n;end
   append!(outputs,Float64.(ap));append!(outputs,Float64.(ao))
  end
  trial==0&&continue
  allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"max_oracle_budget_fraction\":%.12g,\"timing_region\":\"sum18_pairwise_distance_reduction_hostwindows_R10\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),maxerr,maxfraction,join(times,","),path,length(artifact))
 end
 allok||error("PairwiseGaussian fulltile/scalar/actualinput/guards failed");println("PASS")
end
