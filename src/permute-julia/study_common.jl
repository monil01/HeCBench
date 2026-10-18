using CUDA,Printf
function permutation_study(factory)
 raw=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"inputs_u32.bin"))));gold=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_u32.bin"))));G=16;guard=UInt32(0xcafebabe);sentinel=UInt32(0xdeadbeef);R=10;allok=true
 for trial in 0:5
  prefix=UInt32[];outputs=UInt32[];times=Float64[];ok=true;bad=0;offset=0
  for (B,T,C,NH) in ((1,1024,768,12),(2,7,45,5),(2,7,45,5),(1,1,1,1))
   count=3B*T*C;fixture=raw[offset+1:offset+count];ref=gold[offset+1:offset+count];offset+=count
   for bs in (32,64,128,256,512,1024)
    ih=vcat(fill(guard,G),fixture,fill(guard,G));oh=vcat(fill(guard,G),fill(sentinel,count),fill(guard,G));di=CuArray(collect(reinterpret(Float32,ih)));do_=CuArray(collect(reinterpret(Float32,oh)));launch! =factory(view(do_,G+1:G+count),view(di,G+1:G+count),B,T,C,NH,bs)
    if trial==0;launch!();continue;end
    before=collect(reinterpret(UInt32,Array(do_)));ok &=before==oh;CUDA.device_synchronize(blocking=true);start=time_ns();for rep in 1:R;launch!();end;push!(times,(time_ns()-start)*1e-9/R)
    ai=collect(reinterpret(UInt32,Array(di)));ao=collect(reinterpret(UInt32,Array(do_)));ok &=ai==ih;bad+=count_mismatches(ao,vcat(fill(guard,G),ref,fill(guard,G)));append!(prefix,ai);append!(prefix,before);append!(outputs,ao)
   end
  end
  trial==0&&continue
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"artifact_word_mismatches\":%d,\"timing_region\":\"sum24_QKV_permutation_launch_sync_R10\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),bad,join(times,","),path,length(artifact))
 end
 allok||error("ExactQKVpayload/input/guard mismatch");println("PASS")
end
count_mismatches(a,b)=count(a.!=b)
