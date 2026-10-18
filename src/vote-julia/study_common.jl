using CUDA,Printf
include(joinpath(@__DIR__,"case_table.jl"))
function vote_study(factory)
 inputs=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"inputs_u32.bin"))));gold=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_u32.bin"))));G=16;allok=true
 for trial in 0:5
  prefix=UInt32[];outputs=UInt32[];times=Float64[];bad=0;ok=true
  for (op,n,offset,repeat,mode,go,nout) in VOTE_CASES
   if op==2
    initial=UInt8[mode==1&&t%2==0&&f==2 ? 1 : 0 for t in 0:95 for f in 0:2];oh=vcat(fill(UInt8(0xa5),G),initial,fill(UInt8(0xa5),G));out=CuArray(oh);input=nothing;launch! =factory(input,view(out,G+1:G+nout),n,op,repeat);expected=vcat(fill(UInt32(0xa5),G),gold[go+1:go+nout],fill(UInt32(0xa5),G))
   else
    ih=vcat(fill(UInt32(0xcafebabe),G),inputs[offset+1:offset+n],fill(UInt32(0xcafebabe),G));oh=vcat(fill(UInt32(0xcafebabe),G),fill(UInt32(0xdeadbeef),n),fill(UInt32(0xcafebabe),G));input=CuArray(ih);out=CuArray(oh);launch! =factory(view(input,G+1:G+n),view(out,G+1:G+nout),n,op,repeat);expected=vcat(fill(UInt32(0xcafebabe),G),gold[go+1:go+nout],fill(UInt32(0xcafebabe),G))
   end
   if trial==0;launch!();CUDA.device_synchronize(blocking=true);continue;end
   before=Array(out);ok &=before==oh;CUDA.device_synchronize(blocking=true);start=time_ns();launch!();CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9)
   if op!=2;actual=Array(input);ok &=actual==ih;append!(prefix,actual);end
   append!(prefix,UInt32.(before));answer=UInt32.(Array(out));bad+=count(answer.!=expected);append!(outputs,answer)
  end
  trial==0&&continue
  ok &=bad==0;allok &=ok;artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"artifact_word_mismatches\":%d,\"timing_region\":\"sum48_singlelaunch_fullwarp_vote_hostwindows\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),bad,join(times,","),path,length(artifact))
 end
 allok||error("Fullwarp truth/state/input/guardcheck failed");println("PASS")
end
