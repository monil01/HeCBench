using CUDA,Printf
const BM_W=112;const BM_H=40;const BM_G=768;const BM_R=10
bmread(name,::Type{T})where T=collect(reinterpret(T,read(joinpath(@__DIR__,name))))
function bm_study(factory)
 input=bmread("study_input_u8.bin",UInt8);window=bmread("study_window_f32.bin",Float32)
 specs=((:img,UInt8,4480),(:win,Float32,64),(:stacks,UInt16,11520),(:counts,UInt32,768),(:weights,Float32,768),(:group,Float32,786432),(:num,Float32,4480),(:den,Float32,4480),(:out,UInt8,4480))
 bufs=Dict();views=Dict();sentinels=Dict()
 for (name,T,n) in specs
  sentinel=T===UInt8 ? T(165) : T===UInt16 ? T(23063) : T===UInt32 ? T(1234567) : T(1515.25)
  b=CUDA.fill(sentinel,n+32);bufs[name]=b;views[name]=view(b,17:n+16);sentinels[name]=sentinel
 end
 a=(; (name=>views[name]for(name,_,_)in specs)...);copyto!(a.img,input);copyto!(a.win,window);launch=factory(a)
 function reset!()
  fill!(a.stacks,UInt16(0));fill!(a.counts,UInt32(0));fill!(a.weights,0f0);fill!(a.group,0f0);fill!(a.out,UInt8(0))
 end
 function run!()
  fill!(a.num,0f0);fill!(a.den,0f0);launch()
 end
 reset!();run!();CUDA.device_synchronize(blocking=true)
 es=bmread("oracle_stacks_u16.bin",UInt16);ec=bmread("oracle_counts_u32.bin",UInt32);enz=bmread("oracle_nonzero_u32.bin",UInt32);ep=bmread("oracle_post_f32.bin",Float32);en=bmread("oracle_num_f32.bin",Float32);ed=bmread("oracle_den_f32.bin",Float32);eo=bmread("oracle_output_u8.bin",UInt8);allok=true
 for trial in 1:5
  reset!();CUDA.device_synchronize(blocking=true);start=time_ns();for _ in 1:10;run!();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/10
  h=(; (name=>Array(views[name])for(name,_,_)in specs)...);ok=h.img==input&&h.win==window&&h.stacks==es&&h.counts==ec;ma=0.;mr=0.;scaled=0.
  for (got,expected)in ((h.group,ep),(h.num,en),(h.den,ed))
   errors=abs.(Float64.(got).-Float64.(expected));ma=max(ma,maximum(errors));mr=max(mr,maximum(errors./max.(abs.(Float64.(expected)),1.)));scaled=max(scaled,maximum(errors./(.003.+.0003.*abs.(Float64.(expected)))));ok&=all(isfinite,got)&&all(errors.<=.003.+.0003.*abs.(Float64.(expected)))
  end
  nz=Int.(round.(ifelse.(h.weights.>0,1f0./h.weights,0f0)));nze=count(nz.!=Int.(enz));ok&=nze==0;ie=maximum(abs.(Int.(h.out).-Int.(eo)));ok&=ie<=1&&all(h.den.>0)
  art=vcat(Float32.(h.img),h.win,Float32[112,40,64,12,19,8,16,3,625,2.7,2500,10],Float32.(h.stacks),Float32.(h.counts),h.weights,h.group,h.num,h.den,Float32.(h.out))
  for(name,_,n)in specs
   raw=Array(bufs[name]);for j in 1:16;push!(art,Float32(raw[j]),Float32(raw[n+16+j]));ok&=raw[j]==sentinels[name]&&raw[n+16+j]==sentinels[name];end
  end
  allok&=ok;path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"relative_error_floor\":1,\"max_scaled_oracle_error\":%.12g,\"image_max_gray_error\":%d,\"nonzero_count_mismatches\":%d,\"exact_stacks\":%s,\"exact_counts\":%s,\"timing_region\":\"full_BM3D_hardthreshold_pipeline_with_aggregation_resets_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(art),ma,mr,scaled,ie,nze,h.stacks==es ? "true" : "false",h.counts==ec ? "true" : "false",path,length(art))
 end
 allok||error("BM3Dfulloutputs/independentoracle/guards failed");println("PASS")
end
