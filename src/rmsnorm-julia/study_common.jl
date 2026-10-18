using CUDA,Printf
function rms_study(factory)
 fixture=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))))
 allok=true;guard=12345.25f0;G=16;R=20
 for trial in 0:5
  prefix=Float32[];outputs=Float32[];times=Float64[];ok=true;maxerr=0.;fo=0;oo=0
  for (N,H) in [(7,1024),(5,1027)],fixture_id in 0:1
   M=N*H;xh=vcat(fill(guard,G),fixture[fo+1:fo+M],fill(guard,G));gh=vcat(fill(guard,G),fixture[fo+M+1:fo+M+H],fill(guard,G));oh=fill(guard,M+2G);ref=oracle[oo+1:oo+M];fo+=M+H;oo+=M
   for BT in [32,64,128,256,512,1024]
    x=CuArray(xh);g=CuArray(gh);o=CuArray(oh);xview=view(x,G+1:G+M);gview=view(g,G+1:G+H);oview=view(o,G+1:G+M);U=H%4==0 ? 4 : 1
    xv=U==4 ? reinterpret(NTuple{4,VecElement{Float32}},xview) : xview;gv=U==4 ? reinterpret(NTuple{4,VecElement{Float32}},gview) : gview;ov=U==4 ? reinterpret(NTuple{4,VecElement{Float32}},oview) : oview
    launch! =factory(xview,xv,gview,gv,oview,ov,H,N,Val(BT),Val(U))
    if trial==0;launch!();CUDA.device_synchronize(blocking=true);continue;end
    initial=Array(o);ok &=reinterpret(UInt32,initial)==reinterpret(UInt32,oh);CUDA.device_synchronize(blocking=true);start=time_ns()
    for rep in 1:R;launch!();end
    CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/R)
    xx=Array(x);gg=Array(g);out=Array(o);ok &=reinterpret(UInt32,xx)==reinterpret(UInt32,xh)&&reinterpret(UInt32,gg)==reinterpret(UInt32,gh);ok &=all(==(guard),out[1:G])&&all(==(guard),out[end-G+1:end])
    for j in 1:M
     a=Float64(out[G+j]);e=abs(a-ref[j]);maxerr=max(maxerr,e);ok &=isfinite(a)&&e<=1e-5+2e-6*abs(ref[j])
    end
    append!(prefix,xx);append!(prefix,gg);append!(prefix,initial);append!(outputs,out)
   end
  end
  trial==0&&continue
  allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum24_twoscan_rms_kernel_hostwindows_R20\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"internal_repeats\":20,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),maxerr,join(times,","),path,length(artifact))
 end
 allok||error("RMSnorm independentoracle/input/guard check failed");println("PASS")
end
