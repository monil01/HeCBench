const STUDY_SIZES=(2,123,3411,83621,872163,1049603)
const STUDY_REPEATS=1000
function makecase(n)
 input=Int32[87231+mod(73i+19*(i÷257),8192) for i in 0:n-1];input[1]=87231;input[2]=95422;grid=min(4096,cld(n,256));words=((13n)÷64+1)*2
 (CuArray(input),CuArray(UInt64[n]),CUDA.zeros(Int32,grid),CUDA.zeros(Int32,grid),CUDA.zeros(UInt8,1),CUDA.zeros(Int32,1),CUDA.zeros(UInt32,words),CUDA.zeros(Int32,512),CUDA.zeros(UInt32,256),input)
end
function study(run!)
 cases=[makecase(n) for n in STUDY_SIZES]
 for args in cases;run!(args[1:9]...);end;CUDA.synchronize()
 for trial in 1:5
  total=0.;got=UInt32[];wanted=UInt32[];ok=true
  for args in cases
   input,num,lo,hi,bits,minimum,out,template,packtemplate,host=args;n=length(host);grid=length(lo);CUDA.fill!(out,0);CUDA.synchronize();start=time_ns();for repeat in 1:STUDY_REPEATS;run!(args[1:9]...);end;CUDA.synchronize();total+=(time_ns()-start)*1e-9/STUDY_REPEATS
   low=fill(typemax(Int32),grid);high=fill(typemin(Int32),grid);packed=zeros(UInt32,length(out))
   for i in 0:n-1
    group=mod(i÷256,grid)+1;low[group]=min(low[group],host[i+1]);high[group]=max(high[group],host[i+1]);raw=UInt32(host[i+1]-87231);bit=i*13;word=bit÷32;shift=bit%32;packed[word+1]|=raw<<shift
    if shift>19;packed[word+2]|=raw>>(32-shift);end
   end
   append!(got,UInt32[reinterpret(UInt32,Array(minimum)[1]),UInt32(Array(bits)[1])]);append!(got,reinterpret(UInt32,Array(lo)));append!(got,reinterpret(UInt32,Array(hi)));append!(got,Array(out))
   append!(wanted,UInt32[87231,13]);append!(wanted,reinterpret(UInt32,low));append!(wanted,reinterpret(UInt32,high));append!(wanted,packed)
  end
  ok=got==wanted;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"sum_six_sizes_native_full_gpu_config_and_buffered_pack_means\",\"jit_excluded\":true,\"internal_repeats\":1000,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Fullbitstream/config/scanpartialoraclefailed")
 end
end
