@inline xorword(a,b)=(a[1]⊻b[1],a[2]⊻b[2],a[3]⊻b[3],a[4]⊻b[4])
@inline function subword(a,table)
 @inbounds return (table[Int(a[1])+1],table[Int(a[2])+1],table[Int(a[3])+1],table[Int(a[4])+1])
end
@inline function rotateword(a,row,::Val{INV}) where INV
 shift=INV ? -row : row
 return ntuple(i->a[mod(i-1+shift,4)+1],Val(4))
end
@inline function gm(a::UInt8,b::UInt8)
 out=UInt8(0)
 for k in 1:8
  if (b&0x01)==0x01;out⊻=a;end
  high=a&0x80;a<<=1
  if high==0x80;a⊻=0x1b;end
  b>>=1
 end
 return out
end
@inline function mixlane(shared,offset,row,lane,::Val{INV}) where INV
 coeff=INV ? (0x0e,0x0b,0x0d,0x09) : (0x02,0x03,0x01,0x01);v=UInt8(0)
 for k in Int32(0):Int32(3)
  @inbounds v⊻=gm(shared[offset+k+1][lane],coeff[mod(k+4-row,4)+1])
 end
 return v
end
@inline mixword(shared,offset,row,inv)=ntuple(lane->mixlane(shared,offset,row,lane,inv),Val(4))
function study(run!)
 fields=Dict{String,String}()
 for line in readlines(joinpath(@__DIR__,"study_fixture.txt"))
  key,value=split(line);fields[key]=value
 end
 plain=hex2bytes(fields["plain"]);cipher=hex2bytes(fields["cipher"]);roundkey=hex2bytes(fields["roundkey"]);sbox=hex2bytes(fields["sbox"]);inverse=hex2bytes(fields["inverse"])
 dp=CuArray(collect(reinterpret(NTuple{4,UInt8},plain)));dc=CuArray(collect(reinterpret(NTuple{4,UInt8},cipher)));rk=CuArray(collect(reinterpret(NTuple{4,UInt8},roundkey)));st=CuArray(sbox);it=CuArray(inverse);out=CuArray(fill((0x00,0x00,0x00,0x00),1024));template=CuArray(fill((0x00,0x00,0x00,0x00),8))
 run!(out,dp,rk,st,template,Val(false));run!(out,dc,rk,it,template,Val(true));CUDA.synchronize()
 for trial in 1:5
  total=0.;got=UInt8[]
  for (input,table,mode) in ((dp,st,Val(false)),(dc,it,Val(true)))
   CUDA.synchronize();t=time_ns();for k in 1:10;run!(out,input,rk,table,template,mode);end;CUDA.synchronize();total+=(time_ns()-t)*1e-9/10;append!(got,collect(reinterpret(UInt8,Array(out))))
  end
  expected=vcat(cipher,plain);ok=got==expected;err=ok ? 0 : 1;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":8192,\"max_abs_error\":",err,",\"timing_region\":\"sum_encrypt_decrypt_bulk_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint8\",\"count\":8192,\"byte_order\":\"little\"}}")
  ok||error("Full AES128OpenSSLoracle failed")
 end
end
