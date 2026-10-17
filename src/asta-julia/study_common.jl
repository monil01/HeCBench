const STUDY_A=13
const STUDY_B=17
const STUDY_BS=(1,32,65,129,256)
const STUDY_REPEATS=100
function study(run!)
 cases=[]
 for b in STUDY_BS
  host=Float32[Float32(mod(73i+19,65521)-32760)/64 for i in 0:STUDY_A*STUDY_B*b-1];expected=similar(host)
  for row in 0:STUDY_A-1,col in 0:STUDY_B-1,lane in 0:b-1;expected[(col*STUDY_A+row)*b+lane+1]=host[(row*STUDY_B+col)*b+lane+1];end
  flags=Int32[mod(i*STUDY_A,STUDY_A*STUDY_B-1)==i ? 0 : 1 for i in 0:STUDY_A*STUDY_B-2];push!(flags,0)
  push!(cases,(Int32(b),host,expected,flags,CuArray(host),CUDA.zeros(Int32,STUDY_A*STUDY_B),CUDA.zeros(Int32,1),CUDA.zeros(Int32,2)))
 end
 function one!(args)
  b,host,expected,flags,input,finished,head,tmp=args;copyto!(input,host);CUDA.fill!(finished,0);CUDA.fill!(head,0);CUDA.synchronize();t=time_ns();run!(Int32(STUDY_A),Int32(STUDY_B),b,input,finished,head,tmp);CUDA.synchronize();elapsed=(time_ns()-t)*1e-9
  got=Array(input);f=Array(finished);h=Array(head)[1];ok=got==expected&&f==flags&&h==STUDY_A*STUDY_B-1+16
  return got,f,h,ok,elapsed
 end
 for args in cases,repeat in 1:10;one!(args);end
 for trial in 1:5
  total=0.;ok=true;artifact=UInt32[]
  for args in cases
   last=nothing
   for repeat in 1:STUDY_REPEATS
    last=one!(args);ok&=last[4];total+=last[5]/STUDY_REPEATS
   end
   append!(artifact,reinterpret(UInt32,last[1]));append!(artifact,reinterpret(UInt32,last[2]));push!(artifact,UInt32(last[3]))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,artifact),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(artifact),",\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"sum_five_payloads_perlaunch_sync_inplace_transpose_means\",\"jit_excluded\":true,\"internal_repeats\":100,\"all_internal_repeats_checked\":true,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(artifact),",\"byte_order\":\"little\"}}")
  ok||error("ASTAfullpermutation/flag/headfailure")
 end
end
