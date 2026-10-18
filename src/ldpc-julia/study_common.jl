using CUDA,Printf
function ldpc_study(decode)
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==37564&&length(gold)==499320&&sizeof(HElement)==4
 h1=vcat(fill(Int8(90),8),Int8.(prefix[290:301]),fill(Int8(90),8));h2=vcat(fill(Int8(90),8),Int8.(prefix[302:325]),fill(Int8(90),8));p1=vcat(fill(UInt32(0x5a5a5a5a),8),UInt32.(prefix[326:409]),fill(UInt32(0x5a5a5a5a),8));p2=vcat(fill(UInt32(0x5a5a5a5a),8),UInt32.(prefix[410:697]),fill(UInt32(0x5a5a5a5a),8));dc1=CuArray(h1);dc2=CuArray(h2);dp1=CuArray(collect(reinterpret(HElement,p1)));dp2=CuArray(collect(reinterpret(HElement,p2)));count1=view(dc1,9:20);count2=view(dc2,9:32);comp1=view(dp1,9:92);comp2=view(dp2,9:296)
 pos=698;cs=Any[]
 for ci in 1:3
  cw=Int(prefix[pos]);pos+=1;n=cw*2304;initial=vcat(fill(-12345f0,8),Float32.(prefix[pos:pos+n-1]),fill(-12345f0,8));pos+=2n;zero=vcat(fill(-12345f0,8),zeros(Float32,cw*27648),fill(-12345f0,8));hardreset=fill(Int32(-12345),n+16);llrO=CuArray(initial);dtO=CuArray(zero);RO=CuArray(zero);hdO=CuArray(hardreset)
  c=(cw=cw,mcw=cw÷2,initial=initial,zero=zero,hardreset=hardreset,llrO=llrO,dtO=dtO,RO=RO,hdO=hdO,llr=view(llrO,9:n+8),dt=view(dtO,9:cw*27648+8),R=view(RO,9:cw*27648+8),hd=view(hdO,9:n+8),count1=count1,count2=count2,comp1=comp1,comp2=comp2);push!(cs,c)
 end
 @assert pos==37564+1
 reset(c)=(copyto!(c.llrO,c.initial);copyto!(c.dtO,c.zero);copyto!(c.RO,c.zero);copyto!(c.hdO,c.hardreset);nothing)
 for c in cs;reset(c);decode(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(prefix);times=Float64[];ok=true;go=0
  for c in cs
   reset(c);actual=Array(c.llrO);ok&=reinterpret(UInt8,actual)==reinterpret(UInt8,c.initial);CUDA.synchronize();t=time_ns();decode(c);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   data=[Float64.(Array(c.llrO)),Float64.(Array(c.dtO)),Float64.(Array(c.RO)),Float64.(Array(c.hdO)),Float64.(actual),Float64.(Array(dc1)),Float64.(Array(dc2)),Float64.(reinterpret(UInt32,Array(dp1))),Float64.(reinterpret(UInt32,Array(dp2)))]
   for x in data
    ref=gold[go+1:go+length(x)];go+=length(x);ok&=all(isfinite,x)&&reinterpret(UInt8,x)==reinterpret(UInt8,ref);append!(art,x)
   end
  end
  @assert go==499320&&length(art)==536884;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":499320,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum3_full10iteration_QCLDPC_decoder_20kernel_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":536884,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"decoder_iterations\":10,\"kernel_launches\":60}\n",trial,sum(times),ok ? "true" : "false",file,string(times))
 end
end
