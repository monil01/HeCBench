function thermal_study(factory)
 root=@__DIR__;fixture=reinterpret(Float32,read(joinpath(root,"fixture.bin")));oracle=reinterpret(Float64,read(joinpath(root,"oracle.bin")))
 @assert length(fixture)==6600237 && length(oracle)==8800256
 cases=[];off=2
 for ci in 1:Int(fixture[1])
  n,z,steps=Int.(fixture[off:off+2]);q=Tuple(fixture[off+3:off+10]);head=copy(fixture[off:off+10]);off+=11;N=n*n*z;L=N+128;inputs=[copy(fixture[off+(k-1)*L:off+k*L-1]) for k in 1:3];off+=3L
  push!(cases,(n,z,steps,q,head,inputs,factory(n)))
 end
 @assert off==length(fixture)+1
 for trial in 0:5
  prefix=Float64[fixture[1]];computed=Float64[];elapsed=0.;ok=true
  for (n,z,steps,q,head,inputs,launch!)in cases
   append!(prefix,head);base=CuArray.(inputs)
   for k in 1:3;v=Array(base[k]);ok&=reinterpret(UInt32,v)==reinterpret(UInt32,inputs[k]);append!(prefix,v);end
   ia,ib=1,3;a=view(base[1],65:length(base[1])-64);p=view(base[2],65:length(base[2])-64);b=view(base[3],65:length(base[3])-64)
   # Views preserve actual256-bytepayload offset; parentbuffers retain guards.
   CUDA.synchronize(CUDA.context());start=time_ns()
   for it in 1:steps;launch!(a,p,b,z,q);a,b=b,a;ia,ib=ib,ia;end
   CUDA.synchronize(CUDA.context());elapsed+=(time_ns()-start)*1e-9
   final=base[ia];previous=base[ib];selected=isodd(steps) ? final : previous
   for v in (final,previous,selected,base[2]);append!(computed,Array(v));end
   for v in base;CUDA.unsafe_free!(v);end
  end
  err=0.
  @assert length(computed)==length(oracle)
  for i in eachindex(oracle);e=abs(computed[i]-oracle[i]);err=max(err,e);ok&=isfinite(computed[i])&&e<=2e-4+2e-6*abs(oracle[i]);end
  if trial==0;ok||error("WarmupOracleMismatch");continue;end
  append!(prefix,computed);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,prefix)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_original_column_kernel_iteration_loops_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_behavior_known_scientific_defect\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(prefix),err,path,length(prefix))
  ok||error("FullOracleMismatch")
 end
end
