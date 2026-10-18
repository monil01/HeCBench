using CUDA,Printf
function mandel_study(factory)
 oracle=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle_i32.bin"))));cases=[(32,48,0),(32,48,1),(32,48,2),(65,97,100),(1080,1920,100)];G=16;guard=Int32(-1234567);allok=true
 for trial in 0:5
  prefix=Int32[];outputs=Int32[];times=Float64[];ok=true;offset=0;bad=0
  for (rows,cols,cap) in cases
   n=rows*cols;initial=fill(guard,n+2G);params=vcat(fill(guard,G),Int32[rows,cols,cap],fill(guard,G));out=CuArray(initial);p=CuArray(params);launch! =factory(out,p,rows,cols)
   if trial==0;launch!();CUDA.device_synchronize(blocking=true);offset+=n;continue;end
   before=Array(out);actualparams=Array(p);ok &=before==initial&&actualparams==params;CUDA.device_synchronize(blocking=true);start=time_ns();launch!();CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9);answer=Array(out);afterparams=Array(p);ok &=afterparams==params
   bad+=count(!=(guard),answer[1:G])+count(!=(guard),answer[end-G+1:end])+count(answer[G+1:G+n].!=oracle[offset+1:offset+n]);ok &=all(x->0<=x<=cap,answer[G+1:G+n]);append!(prefix,before);append!(prefix,actualparams);append!(outputs,answer);offset+=n
  end
  trial==0&&continue
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum5_source_mandel_kernel_hostwindows\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),bad==0 ? 0 : 1,join(times,","),path,length(artifact))
 end
 allok||error("FullFloat32orbit/input/guard failure");println("PASS")
end
