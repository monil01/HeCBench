function paged_study(uniform!,normal!,attention!)
 root=@__DIR__;fixture=reinterpret(Float64,read(joinpath(root,"fixture.bin")));oracle=reinterpret(Float64,read(joinpath(root,"oracle.bin")))
 guarded(v)=vcat(fill(eltype(v)(-37),32),v,fill(eltype(v)(-37),32))
 for trial in 0:5
  prefix=Float64[3];computed=Float64[];extra=Float64[];off=2;go=1;ok=true;secs=0.;maxerr=0.;times=Float64[]
  for ci in 0:2
   head=copy(fixture[off:off+7]);off+=8;append!(prefix,head)
   S,H,D,maxlen,pages=Int.(head[1:5]);Q=S*H*D;KV=pages*H*D*16;M=cld(maxlen,16);T=S*M;scale=Float32(head[7]);kvscale=Float32(head[8])
   lensoff=off;off+=S;tableoff=off;off+=T;baseline=off;off+=Q+2KV+H;actual=off;off+=Q+2KV+H
   inputs=Any[guarded(zeros(Float32,Q)),guarded(zeros(Float32,KV)),guarded(zeros(Float32,KV)),guarded(zeros(Float32,H)),guarded(zeros(Float32,Q)),guarded(Int32.(fixture[tableoff:tableoff+T-1])),guarded(Int32.(fixture[lensoff:lensoff+S-1]))]
   base=CuArray.(inputs);buffers=map(a->view(a,33:length(a)-32),base);q,k,v,a,out,table,lens=buffers
   for (b,initial)in zip(base,inputs);z=Array(b);ok&=z==initial;append!(prefix,z);end
   uniform!(q,Q,-scale,scale);uniform!(k,KV,-kvscale,kvscale);uniform!(v,KV,-kvscale,kvscale);normal!(a,H);CUDA.synchronize(CUDA.context())
   b=baseline
   for j in 1:4
    z=Array(base[j]);ok&=z[1:32]==inputs[j][1:32]&&z[end-31:end]==inputs[j][end-31:end]
    ok&=Float64.(z[33:end-32])==fixture[b:b+length(z)-65];b+=length(z)-64;append!(prefix,z)
   end
   b=actual
   for j in 1:4
    n=length(buffers[j]);copyto!(buffers[j],Float32.(fixture[b:b+n-1]));b+=n
   end
   before=Array.(base);b=actual
   for j in 1:7
    z=before[j];ok&=z[1:32]==inputs[j][1:32]&&z[end-31:end]==inputs[j][end-31:end]
    if j<=4;n=length(z)-64;ok&=Float64.(z[33:end-32])==fixture[b:b+n-1];b+=n;end
    append!(prefix,z)
   end
   N=D+8+max(M*16,2D)
   CUDA.synchronize(CUDA.context());start=time_ns();attention!(out,q,k,v,table,lens,a,H,S,M,scale,D,N);CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;push!(times,dt);secs+=dt
   z=Array(base[5]);ok&=z[1:32]==inputs[5][1:32]&&z[end-31:end]==inputs[5][end-31:end]
   for i in 1:Q
    e=abs(Float64(z[i+32])-oracle[go+i-1]);maxerr=max(maxerr,e);ok&=isfinite(z[i+32])&&e<=2e-5+2e-5*abs(oracle[go+i-1])
   end
   go+=Q;append!(computed,z)
   for j in 1:7
    z=Array(base[j]);ok&=z[1:32]==inputs[j][1:32]&&z[end-31:end]==inputs[j][end-31:end]
    if j!=5;ok&=z==before[j];end
    append!(extra,z)
   end
   for b in base;CUDA.unsafe_free!(b);end
  end
  @assert off==length(fixture)+1&&go==length(oracle)+1&&length(prefix)==1016993
  if trial==0;ok||error("Pagedwarmupfailed abs=$maxerr");continue;end
  data=vcat(prefix,computed,extra);@assert length(data)==1357921
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,data)
  ok&=success(pipeline(Cmd(["python3",joinpath(root,"audit_native_outputs.py"),path]),stdout=devnull,stderr=stderr))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"case_times_seconds\":[%.17g,%.17g,%.17g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum3source_FP32_pagedattention_launch_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_GPU_body_unchanged\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,times[1],times[2],times[3],ok ? "true" : "false",length(data),maxerr,path,length(data));flush(stdout)
  ok||error("IndependentFullAttentionCertificateFailed")
 end
end
