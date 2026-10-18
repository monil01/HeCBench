using CUDA,Printf
include("encoder_common.jl")
function rle_study(factory)
 io=open(joinpath(@__DIR__,"fixture.bin"));nc=Int(read(io,Int32));cases=[];prefix=Int32[];guard=Int32(1597463007)
 for z in 1:nc
  h=read!(io,Vector{Int32}(undef,4));n=Int(h[1]);kind=h[2];input=CuArray(read!(io,Vector{Int32}(undef,n)))
  if kind==1;append!(prefix,Array(input));else;all(Array(input).==Int32(1))||error("constant_input");end
  buf=zeros(Int32,n+16);buf[n+1:end].=guard;keys=CuArray(buf);lens=CuArray(buf);cnt=fill(guard,17);cnt[1]=0;count=CuArray(cnt);encode=factory(input,keys,lens,count,n);encode();push!(cases,(encode,keys,lens,count))
 end
 close(io);CUDA.device_synchronize(blocking=true);expected=reinterpret(Int32,read(joinpath(@__DIR__,"expected_i32.bin")));allok=true
 for trial in 1:5
  data=copy(prefix);times=Float64[]
  for (encode,keys,lens,count)in cases
   CUDA.device_synchronize(blocking=true);start=time_ns()
   for repeat in 1:1000;encode();CUDA.device_synchronize(blocking=true);end
   push!(times,(time_ns()-start)*1e-9/1000)
   for a in (keys,lens,count);append!(data,Array(a));end
  end
  observed=@view data[length(prefix)+1:end];ok=observed==expected;allok&=ok;bad=Base.count(observed.!=expected);err=maximum(abs.(Int64.(observed).-Int64.(expected)));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"rle_trial$(trial).bin");write(file,data)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"mismatches\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"jit_excluded\":true,\"component_seconds\":[%s],\"timing_region\":\"sum7_source1000Encode_means_each_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(expected),bad,err,join(times,","),file,length(data))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
