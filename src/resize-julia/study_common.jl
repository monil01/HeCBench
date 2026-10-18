using CUDA,Printf
function resize_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==28417&&pre[1]==24&&length(gold)==105568
 pos=2;cases=Any[]
 for path in 1:24
  bits,mode,iw,ih,ow,oh,C,repeat=Int.(pre[pos:pos+7]);pos+=8;T=bits==8 ? UInt8 : bits==16 ? UInt16 : UInt32;ni=C*iw*ih;no=C*ow*oh;h=T.(pre[pos:pos+ni-1]);pos+=ni;lead=256÷sizeof(T);guard=T(bits==8 ? 0xa5 : bits==16 ? 0xa55a : 0xa55aa55a)
  a=CuArray(vcat(fill(guard,lead),h,fill(guard,16)));o=CuArray(fill(guard,lead+no+16));input=view(a,lead+1:lead+ni);output=view(o,lead+1:lead+no)
  @assert UInt64(pointer(input))%256==0&&UInt64(pointer(output))%256==0
  push!(cases,(path=path,bits=bits,mode=Val(mode==1),iw=Int32(iw),ih=Int32(ih),ow=Int32(ow),oh=Int32(oh),C=C,repeat=repeat,ni=ni,no=no,h=h,a=a,o=o,input=input,output=output,lead=lead,guard=guard,fy=Float32(ih)/Float32(oh),fx=Float32(iw)/Float32(ow)))
 end
 @assert pos==length(pre)+1
 for c in cases
  for rep in 1:c.repeat;launch(c);end;CUDA.synchronize()
 end
 for trial in 1:5
  art=copy(pre);times=Float64[];ok=true;ma=0.;gpos=1
  for c in cases
   fill!(c.o,c.guard);CUDA.synchronize();before=Array(c.input);ok &=reinterpret(UInt8,before)==reinterpret(UInt8,c.h)
   t=time_ns();for rep in 1:c.repeat;launch(c);end;CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   h=Array(c.a);o=Array(c.o);ok &=reinterpret(UInt8,h[c.lead+1:c.lead+c.ni])==reinterpret(UInt8,c.h)
   guards=vcat(h[1:c.lead],h[end-15:end],o[1:c.lead],o[end-15:end]);ok &=all(==(c.guard),guards)
   vals=vcat(Float64.(o[c.lead+1:c.lead+c.no]),Float64.(h[c.lead+1:c.lead+c.ni]),Float64.(guards));ref=gold[gpos:gpos+length(vals)-1];gpos+=length(vals);ok &=vals==ref;ma=max(ma,maximum(abs.(vals.-ref)));append!(art,vals)
  end
  @assert gpos==length(gold)+1;f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum24_typed_nearest_bilinear_corrected_reference_paths_3repeats\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"source_grid_blocks\":29184,\"source_threads_per_block\":256,\"native_reference_repaired\":true}\n",trial,sum(times),ok ? "true" : "false",length(art),ma,f,length(art),string(times))
 end
end
