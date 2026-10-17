using CUDA,Printf,LinearAlgebra
function fixture(f)
 n=(32,48,64)[f];a=zeros(Float32,n*n)
 if f==1
  for i in 0:n-1,j in 0:n-1;a[i*n+j+1]=10f0*ccall(:expf,Cfloat,(Cfloat,),-.001f0*Float32(abs(i-j)));end
 elseif f==2
  for i in 0:n-1,j in 0:n-1;a[i*n+j+1]=i==j ? Float32(n+1) : Float32((13i+7j)%17-8)/32;end
 else
  l=Matrix{Float64}(I,n,n);u=zeros(n,n)
  for i in 0:n-1,j in 0:n-1
   if i>j;l[i+1,j+1]=((3i+5j)%9-4)/64;end
   if i==j;u[i+1,j+1]=2+i/64;elseif j>i;u[i+1,j+1]=((7i+2j)%11-5)/32;end
  end
  a=Float32.(vec(transpose(l*u)))
 end
 m=Matrix(transpose(reshape(Float64.(a),n,n)));gold=lu(m,NoPivot();check=true).factors
 if f==3
  @assert maximum(abs.(gold.-(l+u-Matrix{Float64}(I,n,n))))<1e-12
 end
 (n,a,m,vec(transpose(gold)))
end
function accuracy(g,f)
 n,a,m,gold=f;v=Matrix(transpose(reshape(Float64.(g),n,n)));l=Matrix(LowerTriangular(v));for i in 1:n;l[i,i]=1;end;u=Matrix(UpperTriangular(v));delta=l*u-m
 residual=maximum(abs.(delta));backward=opnorm(delta,Inf)/opnorm(m,Inf)
 xstar=Float64[(i%7-3)/8 for i in 0:n-1];rhs=m*xstar;xhat=UpperTriangular(u)\(LowerTriangular(l)\rhs);forward=norm(xhat-xstar,Inf)/norm(xstar,Inf)
 err=maximum(abs.(Float64.(g).-gold));ok=all(isfinite,g)&&all(abs.(Float64.(g).-gold).<=1e-3.+1e-5.*abs.(gold))&&residual<=1e-4&&backward<=1e-6
 (ok,err,residual,backward,forward,cond(m,Inf))
end
function study_main(launch!)
 rep=isempty(ARGS) ? 1 : parse(Int,ARGS[1]);rep==1||error("inplaceLU uses exactlyonefactorization perreset trial");fs=[fixture(f) for f in 1:3];ds=[CuArray(f[2]) for f in fs];prefix=vcat([vcat(Float32[f[1]],f[2]) for f in fs]...)
 for (f,d) in zip(fs,ds);launch!(d,f[1]);end;CUDA.synchronize()
 for trial in 1:5
  for (f,d) in zip(fs,ds);copyto!(d,f[2]);end;CUDA.synchronize();st=time_ns();for (f,d) in zip(fs,ds);launch!(d,f[1]);end;CUDA.synchronize();elapsed=(time_ns()-st)*1e-9
  got=Float32[];ok=true;err=0.;checks=[]
  for (f,d) in zip(fs,ds)
   g=Array(d);metric=accuracy(g,f);ok&=metric[1];err=max(err,metric[2]);push!(checks,metric);append!(got,g)
  end
  for (i,c) in enumerate(checks);@printf("LU_ACCURACY trial=%d case=%d factor_abs=%.17g residual_abs=%.17g backward_inf=%.17g solve_forward_rel=%.17g condition_inf=%.17g\n",trial,i,c[2:end]...);end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"three_complete_nopivot_blocked_LU_factorizations_kernelonly_hostdispatch_sync\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,path,length(art));ok||error("LUfactors/reconstructioncontract; forwarderrorseparatedfrombackwarderror")
 end
end
