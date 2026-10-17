include(joinpath(@__DIR__,"study_fixture.jl"))
const BLOCKS_COUNT=1025
@inline function update_key(key,round,sbox)
 input=round==Int32(1) ? key : Base.setindex(key,key[6]⊻UInt8(round<<Int32(2)),6)
 rotated=ntuple(i->(input[mod(i-1+7,10)+1]<<5)|(input[mod(i-1+8,10)+1]>>3),Val(10))
 @inbounds high=sbox[Int(rotated[1]>>4)+1]
 rotated=Base.setindex(rotated,(rotated[1]&0x0f)|high,1)
 if round==Int32(1)
  rotated=Base.setindex(rotated,rotated[8]⊻UInt8(round>>1),8);rotated=Base.setindex(rotated,rotated[9]⊻UInt8(round<<7),9)
 end
 return rotated
end
@inline function combine(state,offset,a,b,c,d)
 @inbounds return (a[Int(state[offset+1])+1]&0xc0)|(b[Int(state[offset+2])+1]&0x30)|(c[Int(state[offset+3])+1]&0x0c)|(d[Int(state[offset+4])+1]&0x03)
end
@inline permute(state,t0,t1,t2,t3)=(combine(state,0,t3,t2,t1,t0),combine(state,4,t3,t2,t1,t0),combine(state,0,t0,t3,t2,t1),combine(state,4,t0,t3,t2,t1),combine(state,0,t1,t0,t3,t2),combine(state,4,t1,t0,t3,t2),combine(state,0,t2,t1,t0,t3),combine(state,4,t2,t1,t0,t3))
@inline load_key(keys,base)=ntuple(i->keys[base+i],Val(10))
@inline load_state(bytes,base,key)=ntuple(i->bytes[base+i]⊻key[i],Val(8))
@inline function encrypt_body(gid,num,rounds,plains,keys,ciphers,sbox,t0,t1,t2,t3)
 if gid<num
  @inbounds key=load_key(keys,gid*Int32(10));state=load_state(plains,gid*Int32(8),key)
  key=update_key(key,Int32(1),sbox);cipher=permute(state,t0,t1,t2,t3)
  for lane in 1:8;@inbounds ciphers[gid*Int32(8)+lane]=cipher[lane];end
  for round in Int32(2):rounds
   @inbounds state=load_state(ciphers,gid*Int32(8),key);cipher=permute(state,t0,t1,t2,t3)
   for lane in 1:8;@inbounds ciphers[gid*Int32(8)+lane]=cipher[lane];end
   key=update_key(key,round,sbox)
  end
  if rounds==Int32(31)
   for lane in 1:8;@inbounds ciphers[gid*Int32(8)+lane]⊻=key[lane];end
  end
 end
 return
end
function study(run!)
 plain=UInt8[];key=UInt8[]
 for block in 0:BLOCKS_COUNT-1
  append!(plain,UInt8[block<4 ? (block>=2 ? 0xff : 0x00) : UInt8(mod(37*block+19*lane+11,256)) for lane in 0:7]);append!(key,UInt8[block<4 ? (isodd(block) ? 0xff : 0x00) : UInt8(mod(53*block+23*lane+7+(block>>8),256)) for lane in 0:9])
 end
 P=CuArray(plain);K=CuArray(key);C=CUDA.zeros(UInt8,8BLOCKS_COUNT);S=CuArray(STUDY_SBOX);T0=CuArray(STUDY_SBOX_PMT_0);T1=CuArray(STUDY_SBOX_PMT_1);T2=CuArray(STUDY_SBOX_PMT_2);T3=CuArray(STUDY_SBOX_PMT_3);args=(Int32(BLOCKS_COUNT),Int32(31),P,K,C,S,T0,T1,T2,T3);run!(args...);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns();for repeat in 1:10;run!(args...);CUDA.synchronize();end;elapsed=(time_ns()-start)*1e-9/10;got=Array(C);ok=got==STUDY_EXPECTED;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":8200,\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"present80_full31rounds_host_sync_kernel_mean\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint8\",\"count\":8200,\"byte_order\":\"little\"}}")
  ok||error("PRESENTindependentspec/fullpublishedKATfailed")
 end
end
