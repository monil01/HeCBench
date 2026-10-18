using CUDA,Printf
const ROOT=@__DIR__
const SENTINEL=(UInt8(0xa5),UInt8(0x5a),UInt8(0xc3))
const LAMBDA=parse(Float32,get(ENV,"STUDY_LAMBDA","1"))
const LABEL=LAMBDA==0 ? "0" : LAMBDA==1 ? "1" : "half"
pixelbytes(a)=collect(reinterpret(UInt8,vec(Array(a))))
function dpid_study(factory)
 prepared=map(enumerate(((40,30),(128,96)))) do item
  ci,(iw,ih)=item;raw=read(joinpath(ROOT,"study_input$(ci)_u8.bin"));host=reshape(collect(reinterpret(NTuple{3,UInt8},raw)),iw,ih);p=Params(16,12,iw,ih,Float32(iw/16),Float32(ih/12),LAMBDA,10)
  ib=CUDA.fill(SENTINEL,iw*ih+32);input=reshape(view(ib,17:16+iw*ih),iw,ih);copyto!(input,host);gb=CUDA.fill(SENTINEL,224);ob=CUDA.fill(SENTINEL,224);g=reshape(view(gb,17:208),16,12);o=reshape(view(ob,17:208),16,12)
  expected=vcat(read(joinpath(ROOT,"oracle_guidance$(ci)_u8.bin")),read(joinpath(ROOT,"oracle_output$(ci)_lambda_$(LABEL)_u8.bin")))
  (p,input,g,o,(ib,gb,ob),factory(input,g,o,p),expected)
 end
 for z in prepared;z[6]();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  prefix=UInt32[];outputs=UInt32[];guards=UInt32[];times=zeros(2);ok=true;err=0
  for (ci,z) in enumerate(prepared)
   p,input,g,o,bases,run,expected=z;fill!(bases[2],SENTINEL);fill!(bases[3],SENTINEL);run();CUDA.device_synchronize(blocking=true)
   start=time_ns();for _ in 1:10;run();end;CUDA.device_synchronize(blocking=true);times[ci]=(time_ns()-start)*1e-9/10
   data=pixelbytes(input);append!(prefix,UInt32.(data));append!(prefix,UInt32[p.o_width,p.o_height,p.i_width,p.i_height]);append!(prefix,reinterpret(UInt32,Float32[p.p_width,p.p_height,p.lambda]));push!(prefix,p.repeat)
   actual=vcat(pixelbytes(g),pixelbytes(o));errors=abs.(Int.(actual).-Int.(expected));err=max(err,maximum(errors));ok&=all(errors.<=1);append!(outputs,UInt32.(actual))
   for base in bases
    raw=Array(base);guard=vcat(raw[1:16],raw[end-15:end]);ok&=all(==(SENTINEL),guard);append!(guards,UInt32.(reinterpret(UInt8,guard)))
   end
  end
  allok&=ok;artifact=vcat(prefix,outputs,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":2880,\"max_abs_error\":%d,\"max_rel_error\":null,\"timing_region\":\"sum_two_patch_regime_guidance_plus_downsampling_launch_loop_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"lambda\":%.9g,\"timed_cases\":2,\"component_time_seconds\":{\"fractional_2p5_patch\":%.12g,\"integer_8_patch\":%.12g},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",err,p.lambda,times[1],times[2],path,length(artifact))
 end
 allok||error("DPIDfullRGBguidance/output/oracle/guards mismatch");println("PASS")
end
