const BLOCKSIZE_BYTES=UInt64(1024*1024)
const BLOCKSIZE_U32=BLOCKSIZE_BYTES÷UInt64(4)
const STUDY_BYTES=2*1024*1024
const STUDY_WORDS=STUDY_BYTES÷4
const STUDY_THREADS=64
const STUDY_BLOCKS=64*1024
const STUDY_REPEATS=3
@inline function record_error5!(buf,count,addr,expect,current,second,index,wanted,got)
 slot=Int(CUDA.atomic_add!(pointer(count),UInt32(1))%UInt32(10))+1
 @inbounds addr[slot]=UInt64(pointer(buf,Int(index)+1));expect[slot]=wanted;current[slot]=got;second[slot]=UInt64(buf[Int(index)+1])
 return
end
function study(run!)
 buf=CUDA.zeros(UInt32,STUDY_WORDS);count=CUDA.zeros(UInt32,1);addr=CUDA.zeros(UInt64,10);expect=similar(addr);current=similar(addr);second=similar(addr)
 function pipeline!()
  run!(:init,buf,UInt64(STUDY_BYTES÷64))
  run!(:move,buf,UInt64(STUDY_BYTES)÷BLOCKSIZE_BYTES)
  run!(:check,buf,UInt64(STUDY_BYTES÷8),count,addr,expect,current,second)
 end
 pipeline!();CUDA.synchronize();original=Array(buf)[1:2];bad=original[1]⊻UInt32(1);copyto!(buf,1,UInt32[bad],1,1);CUDA.fill!(count,0)
 run!(:check,buf,UInt64(STUDY_BYTES÷8),count,addr,expect,current,second);CUDA.synchronize()
 injection_ok=Array(count)[1]==1&&Array(addr)[1]==UInt64(pointer(buf))&&Array(expect)[1]==original[2]&&Array(current)[1]==bad&&Array(second)[1]==bad
 injection_ok||error("Injected pair fault/errorrecordcheckfailed")
 expected=Vector{UInt32}(undef,STUDY_WORDS);half=Int(BLOCKSIZE_U32÷2)
 for i in 0:STUDY_WORDS-1
  block=i÷Int(BLOCKSIZE_U32);pos=i%Int(BLOCKSIZE_U32);src=pos<half ? mod(pos-8,half) : pos-half;absolute=block*Int(BLOCKSIZE_U32)+src;p=UInt32(1)<<((absolute÷16)%32);expected[i+1]=absolute%4<2 ? p : ~p
 end
 for trial in 1:5
  CUDA.fill!(count,0);CUDA.synchronize();start=time_ns()
  for repeat in 1:STUDY_REPEATS;pipeline!();end
  CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/STUDY_REPEATS;got=Array(buf);errors=Array(count)[1];ok=got==expected&&errors==0&&injection_ok
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(path,"w")do io;write(io,got);write(io,errors);end
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":524289,\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"native_test5_init_move_check_sync_chain_mean\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":524289,\"byte_order\":\"little\"}}")
  ok||error("Fullbufferanderrorcountfailed")
 end
end
