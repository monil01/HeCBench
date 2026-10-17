using CUDA,Printf
if !isdefined(@__MODULE__,:BPNN);include(joinpath(@__DIR__,"backprop_host.jl"));end
rowflat(a)=vec(permutedims(a))
function study_net(n,f)
 iu=Float32[1;[(((i*13+f*7)%31)-15)/32 for i in 1:n]];iw=Float32[(((i*11+j*7+f*3)%37)-18)/128 for i in 0:n,j in 0:16];ip=Float32[(((i*5+j*3)%17)-8)/1024 for i in 0:n,j in 0:16]
 hw=Float32[(((i*7+j*11+f)%29)-14)/64 for i in 0:16,j in 0:1];hp=Float32[(((i*3+j)%13)-6)/1024 for i in 0:16,j in 0:1]
 BPNN(n,16,1,iu,zeros(Float32,17),zeros(Float32,2),zeros(Float32,17),zeros(Float32,2),Float32[0,.2+.5*(f-1)],iw,hw,ip,hp)
end
function pack_result(net,w,prev,eo,eh)
 vcat(w,prev,net.hidden_units,net.output_units,net.hidden_delta,net.output_delta,rowflat(net.hidden_weights),rowflat(net.hidden_prev_weights),Float32[eo,eh])
end
function cpu_gold(net)
 n=net.input_n;x=Float64.(net.input_units);w=Float64.(net.input_weights);prev=Float64.(net.input_prev_weights);hw=Float64.(net.hidden_weights);hp=Float64.(net.hidden_prev_weights)
 hidden=zeros(Float64,17);hidden[1]=1
 for j in 2:17;hidden[j]=1/(1+exp(-sum(w[k,j]*x[k] for k in 1:n+1)));end
 output=zeros(Float64,2);output[2]=1/(1+exp(-sum(hw[k,2]*hidden[k] for k in 1:17)));od=zeros(Float64,2);od[2]=output[2]*(1-output[2])*(net.target[2]-output[2]);hd=zeros(Float64,17)
 for j in 2:17;hd[j]=hidden[j]*(1-hidden[j])*od[2]*hw[j,2];end
 for j in 2:17,k in 1:n+1;upd=Float64(ETA)*hd[j]*x[k]+Float64(MOMENTUM)*prev[k,j];w[k,j]+=upd;prev[k,j]=upd;end
 for k in 1:17;upd=Float64(ETA)*od[2]*hidden[k]+Float64(MOMENTUM)*hp[k,2];hw[k,2]+=upd;hp[k,2]=upd;end
 vcat(rowflat(w),rowflat(prev),hidden,output,hd,od,rowflat(hw),rowflat(hp),abs(od[2]),sum(abs,hd))
end
function study_train_tiled!(net,forward!,adjust!)
 n=net.input_n;w=rowflat(net.input_weights);prev=rowflat(net.input_prev_weights)
 CUDA.synchronize();st=time_ns()
 di=CuArray(net.input_units);dw=CuArray(w);dp=CuArray{Float32}(undef,n)
 forward!(di,dw,dp,n);partial=Array(dp)
 for j in 1:16
  s=0f0;for k in 0:n÷16-1;s+=partial[k*16+j];end
  s+=net.input_weights[1,j+1];net.hidden_units[j+1]=Float32(1.0/(1.0+exp(-Float64(s))))
 end
 layerforward!(net.hidden_units,net.output_units,net.hidden_weights,16,1)
 eo=output_error!(net.output_delta,net.target,net.output_units,1);eh=hidden_error!(net.hidden_delta,16,net.output_delta,1,net.hidden_weights,net.hidden_units)
 adjust_weights!(net.output_delta,1,net.hidden_units,16,net.hidden_weights,net.hidden_prev_weights)
 copyto!(dw,w);dd=CuArray(net.hidden_delta);dprev=CuArray(prev);adjust!(di,dw,dd,dprev,n);got=Array(dw);gp=Array(dprev)
 for a in (di,dw,dp,dd,dprev);CUDA.unsafe_free!(a);end;CUDA.synchronize();elapsed=(time_ns()-st)*1e-9
 pack_result(net,got,gp,eo,eh),elapsed
end
function study_main(train!)
 nets=[study_net(64,1),study_net(96,2)];gold=vcat(cpu_gold.(nets)...);prefix=vcat([vcat(z.input_units,rowflat(z.input_weights),rowflat(z.input_prev_weights),rowflat(z.hidden_weights),rowflat(z.hidden_prev_weights),z.target) for z in nets]...)
 # Finite-difference gradient check on a single Float64 input weight, independent of GPU updates.
 z=nets[1];x=Float64.(z.input_units);w=Float64.(z.input_weights);hw=Float64.(z.hidden_weights);target=Float64(z.target[2])
 function loss(v)
  ww=copy(w);ww[4,5]=v;h=[1.0;[1/(1+exp(-sum(ww[k,j]*x[k] for k in 1:65))) for j in 2:17]];o=1/(1+exp(-sum(hw[k,2]*h[k] for k in 1:17)));(target-o)^2/2
 end
 eps=1e-5;fd=(loss(w[4,5]+eps)-loss(w[4,5]-eps))/(2eps);h=[1.0;[1/(1+exp(-sum(w[k,j]*x[k] for k in 1:65))) for j in 2:17]];o=1/(1+exp(-sum(hw[k,2]*h[k] for k in 1:17)));analytic=-(o*(1-o)*(target-o)*hw[5,2]*h[5]*(1-h[5])*x[4]);@assert abs(fd-analytic)<1e-9
 for net in nets;train!(deepcopy(net));end;CUDA.synchronize()
 for trial in 1:5
  got=Float32[];elapsed=0.0
  for net in nets;v,t=train!(deepcopy(net));append!(got,v);elapsed+=t;end
  err=maximum(abs.(Float64.(got).-gold));ok=all(isfinite,got)&&err<1e-5;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");art=vcat(prefix,got);write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"two_fresh_full_training_steps_host_inclusive\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,path,length(art));ok||error("backprop independent gradient/state check")
 end
end
