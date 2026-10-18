using CUDA,Printf
encode_words(x::AbstractArray{Int32})=Float64.(x)
function encode_words(x::AbstractArray{UInt64})
 a=Float64[];for v in x;push!(a,Float64(v&0xffffffff));push!(a,Float64(v>>32));end;a
end
function metropolis_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));pos=2;cs=Any[]
 for ci in 1:Int(pre[1])
  L=Int(pre[pos]);h=Float32(pre[pos+1]);B=Float32(pre[pos+2]);ns=Int(pre[pos+3]);pos+=4;@assert L==32;hs=Any[];ds=Any[];vs=Any[]
  for (n,T) in [(32768,Int32),(32768,Int32),(8192,UInt64),(8192,UInt64)]
   a=if T==Int32;x=Int32.(pre[pos:pos+n-1]);pos+=n;x
   else;x=UInt64[UInt64(pre[pos+2*i])|(UInt64(pre[pos+2*i+1])<<32) for i in 0:n-1];pos+=2*n;x;end
   g=T==Int32 ? T(-12345) : UInt64(0x5a5a5a5a5a5a5a5a);hh=vcat(fill(g,8),a,fill(g,8));d=CuArray(hh);push!(hs,hh);push!(ds,d);push!(vs,view(d,9:n+8))
  end
  push!(cs,(h=h,B=B,ns=ns,hs=hs,ds=ds,vs=vs))
 end
 @assert pos==length(pre)+1
 for c in cs;launch(c);end
 for trial in 1:5
  art=copy(pre);ok=true;go=0;times=Float64[]
  for c in cs
   for (d,h) in zip(c.ds,c.hs);copyto!(d,h);end
   CUDA.synchronize();before=[Array(d) for d in c.ds];ok&=all(reinterpret(UInt8,a)==reinterpret(UInt8,b) for (a,b) in zip(before,c.hs));t=time_ns();launch(c);push!(times,(time_ns()-t)*1e-9)
   after=[Array(d) for d in c.ds]
   for group in [after,before],arr in group
    v=encode_words(arr);ref=gold[go+1:go+length(v)];go+=length(v);ok&=v==ref;append!(art,v)
   end
  end
  @assert go==length(gold);f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum3_complete_corrected_Ising_Metropolis_sweep_components\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",go,f,length(art),string(times))
 end
end
