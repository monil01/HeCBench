const TT=4;const HH=72;const RR=16;const LL=3
@inline function partial(tid,token,channel,input,weights,indices,::Val{BS},::Val{MODE}) where {BS,MODE}
 @inbounds lid=indices[token+1];acc=0f0
 if MODE==1
  k=tid
  while k<Int32(HH)
   @inbounds acc+=Float32(input[token*Int32(HH)+k+1])*Float32(weights[(lid*Int32(RR)+channel)*Int32(HH)+k+1]);k+=Int32(BS)
  end
 elseif MODE==2
  k=tid
  while k<Int32(HH÷8)
   @inbounds a=input[token*Int32(HH÷8)+k+1];w=weights[(lid*Int32(RR)+channel)*Int32(HH÷8)+k+1]
   for lane in 1:8;acc+=Float32(a[lane])*Float32(w[lane]);end
   k+=Int32(BS)
  end
 else
  k=tid
  while k<Int32(RR)
   @inbounds acc+=input[token*Int32(RR)+k+1]*Float32(weights[(lid*Int32(HH)+channel)*Int32(RR)+k+1]);k+=Int32(BS)
  end
 end
 return acc
end
@inline function store_result(output,token,channel,sum,::Val{MODE}) where MODE
 idx=token*Int32(MODE<=2 ? RR : HH)+channel+1
 if MODE<=2;@inbounds output[idx]+=0.5f0*sum
 else
  @inbounds prev=MODE==4 ? Float32(output[idx]) : 0f0
  @inbounds output[idx]=Float16(prev+sum)
 end
 return
end
function study(run!)
 indices=CuArray(Int32[mod(2*t,LL) for t in 0:TT-1]);a=Float16[Float16(mod(i,17)-8)/Float16(32) for i in 0:TT*HH-1];w=Float16[Float16(mod(i,13)-6)/Float16(32) for i in 0:LL*RR*HH-1];x=Float32[Float32(mod(i,11)-5)/32f0 for i in 0:TT*RR-1];v=Float16[Float16(mod(i,7)-3)/Float16(32) for i in 0:LL*HH*RR-1];A=CuArray(a);W=CuArray(w);AV=CuArray(collect(reinterpret(NTuple{8,Float16},a)));WV=CuArray(collect(reinterpret(NTuple{8,Float16},w)));X=CuArray(x);V=CuArray(v);Y=CUDA.zeros(Float32,TT*RR);Z=CUDA.zeros(Float16,TT*HH);templates=Dict(bs=>CUDA.zeros(Float32,bs) for bs in (32,64,128,256,512,1024));cases=[]
 for bs in (32,64,128,256,512,1024),mode in 1:4
  input=mode==1 ? A : mode==2 ? AV : X;weights=mode==1 ? W : mode==2 ? WV : V;out=mode<=2 ? Y : Z;channels=mode<=2 ? RR : HH;want=Float32[]
  for t in 0:TT-1,ch in 0:channels-1
   lid=mod(2*t,LL);numerator=mode<=2 ? sum((mod(t*HH+k,17)-8)*(mod((lid*RR+ch)*HH+k,13)-6) for k in 0:HH-1) : sum((mod(t*RR+k,11)-5)*(mod((lid*HH+ch)*RR+k,7)-3) for k in 0:RR-1);dot=Float32(numerator)/1024f0;value=0f0
   for repeat in 1:10
    value=mode<=2 ? value+0.5f0*dot : Float32(Float16((mode==4 ? value : 0f0)+dot))
   end
   push!(want,value)
  end
  push!(cases,(bs,mode,input,weights,out,templates[bs],want))
 end
 for (bs,mode,input,weights,out,template,want) in cases
  fill!(out,zero(eltype(out)));run!(out,input,weights,indices,template,Val(bs),Val(mode))
 end
 CUDA.synchronize()
 for trial in 1:5
  total=0.;got=Float32[];wantall=Float32[]
  for (bs,mode,input,weights,out,template,want) in cases
   fill!(out,zero(eltype(out)));CUDA.synchronize();start=time_ns();for repeat in 1:10;run!(out,input,weights,indices,template,Val(bs),Val(mode));end;CUDA.synchronize();total+=(time_ns()-start)*1e-9/10;append!(got,Float32.(Array(out)));append!(wantall,want)
  end
  err=maximum(abs.(got.-wantall));ok=all(isfinite,got)&&err<.1;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":4224,\"max_abs_error\":",err,",\"timing_region\":\"sum_six_blocks_four_modes_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":4224,\"byte_order\":\"little\"}}")
  ok||error("BGMVintegerdot+halfrecurrenceoracle failed")
 end
end
