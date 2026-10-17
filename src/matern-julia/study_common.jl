using CUDA,Printf,LinearAlgebra
function matern_fixture()
 n=4913;targets=Float32[]
 for i=0:n-1;append!(targets,Float32[i%17,(i÷17)%17,i÷289]./16f0);end
 sources=vcat((targets[3*((73*j+17)%n)+1:3*((73*j+17)%n)+3] for j=0:49)...)
 positive=Float32[(j%7+1)/16 for j=0:49];signed=Float32[(isodd(j) ? -1 : 1)*(j%7+1)/16 for j=0:49];scales=Float32[.1,1,10,100,1000,10000,100000]
 return sources,targets,[positive,signed],scales
end
function matern_oracle(s,t,weights,l,::Type{T}) where T
 out=zeros(T,length(t)÷3)
 for i=1:length(out),j=1:50
  if T==Float32
   q=0f0
   for k=0:2;v=s[3*j-2+k]-t[3*i-2+k];q+=v*v;end
   diff=sqrt(q);out[i]+=(1f0+sqrt(5f0)*diff/l+5f0*q/(3f0*l*l))*exp(-sqrt(5f0)*diff/l)*weights[j]
  else
   q=sum((Float64(s[3*j-2+k])-Float64(t[3*i-2+k]))^2 for k=0:2);a=sqrt(5.)*sqrt(q)/Float64(l);out[i]+=(1+a+a*a/3)*exp(-a)*Float64(weights[j])
  end
 end
 return out
end
function matern_study(launch!)
 s,t,w,scales=matern_fixture();n=length(t)÷3;prefix=vcat(s,t,w...,scales);expected=Float32[];gold=Float64[];states=[];ds=CuArray(s);dt=CuArray(t)
 for mode=1:2
  dw=CuArray(w[mode])
  for l in scales
   append!(expected,matern_oracle(s,t,w[mode],l,Float32));append!(gold,matern_oracle(s,t,w[mode],l,Float64));out=CUDA.zeros(Float32,n);push!(states,(l,dw,out));launch!(Int32(n),l,ds,dt,dw,out)
  end
 end
 for l in (scales[1],scales[2],scales[7])
  gram=zeros(Float64,50,50)
  for i=1:50,j=1:50
   r=sqrt(sum((Float64(s[3*i-2+k])-Float64(s[3*j-2+k]))^2 for k=0:2));a=sqrt(5.)*r/Float64(l);gram[i,j]=(1+a+a*a/3)*exp(-a)
  end
  spectrum=eigvals(Symmetric(gram));minimum(spectrum)>=-1e-10*maximum(spectrum)||error("Covariance PSD sanity check");maximum(abs.(diag(gram).-1))==0||error("Zero-distance normalization")
  @printf("MATERN_GRAM l=%.9g eigen_min=%.12g eigen_max=%.12g\n",l,minimum(spectrum),maximum(spectrum))
 end
 CUDA.synchronize();allok=true
 for trial=1:5
  for (_,_,out) in states;fill!(out,Float32(NaN));end
  CUDA.synchronize();start=time_ns()
  for rep=1:5,(l,dw,out) in states;launch!(Int32(n),l,ds,dt,dw,out);end
  CUDA.synchronize();sec=(time_ns()-start)*1e-9/5;vals=vcat((Array(out) for (_,_,out) in states)...);err=maximum(abs.(vals.-expected));science=maximum(abs.(Float64.(vals).-gold));positive=reshape(vals[1:7*n],n,7);monotonic=minimum(diff(positive,dims=2))>=-1e-4;bounds=minimum(positive)>=-1e-6&&maximum(positive)<=sum(w[1])+1e-4;ok=all(isfinite,vals)&&err<=1e-3&&science<=1e-4&&monotonic&&bounds;allok&=ok
  @printf("MATERN_SCIENCE trial=%d FP64_error=%.12g sourceCPU_error=%.12g\n",trial,science,err)
  out=vcat(prefix,vals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":68782,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"independent_Float64_error_inf\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":5,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",err,science,path,length(out))
 end
 allok||error("Matérn full-output mathematical mismatch");println("PASS")
end
