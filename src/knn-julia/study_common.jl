using CUDA,Printf
function fixture(f)
 n,q,d,k=f==1 ? (129,67,68,20) : (65,37,5,7);a=zeros(Float32,n*d);b=zeros(Float32,q*d)
 if f==1
  ccall(:srand,Cvoid,(Cuint,),2)
  for x in (a,b),i in eachindex(x);x[i]=Float32(ccall(:rand,Cint,()))/Float32(typemax(Cint));end
 else
  for z in 0:d-1,i in 0:n-1;a[z*n+i+1]=Float32(((i%13)*3+z*5)%17)/8-.5f0;end
  for z in 0:d-1,i in 0:q-1;b[z*q+i+1]=Float32(((i%11)*7+z*3)%19)/8-.5f0;end
 end
 gd=zeros(Float64,q*k);gi=zeros(Int32,q*k)
 for j in 0:q-1
  ss=[sum((Float64(a[z*n+i+1])-Float64(b[z*q+j+1]))^2 for z in 0:d-1) for i in 0:n-1];ix=sortperm(0:n-1;by=i->(ss[i+1],i))
  for t in 0:k-1;gi[t*q+j+1]=ix[t+1]-1;gd[t*q+j+1]=sqrt(ss[ix[t+1]]);end
 end
 (n,q,d,k,a,b,gd,gi)
end
function study_main(knn_call!)
 rep=isempty(ARGS) ? 10 : parse(Int,ARGS[1]);fs=[fixture(f) for f in 1:2];outs=[(zeros(Float32,f[2]*f[4]),zeros(Int32,f[2]*f[4])) for f in fs];prefix=vcat([vcat(Float64[f[1],f[2],f[3],f[4]],Float64.(f[5]),Float64.(f[6])) for f in fs]...)
 for (f,o) in zip(fs,outs);knn_call!(f[5],f[6],f[1],f[2],f[3],f[4],o[1],o[2]);end;CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();st=time_ns()
  for _ in 1:rep,(f,o) in zip(fs,outs);knn_call!(f[5],f[6],f[1],f[2],f[3],f[4],o[1],o[2]);end
  CUDA.synchronize();elapsed=(time_ns()-st)*1e-9/rep;got=Float64[];err=0.;ok=true
  for (f,o) in zip(fs,outs)
   delta=maximum(abs.(Float64.(o[1]).-f[7]));err=max(err,delta);ok&=all(isfinite,o[1])&&delta<=1e-3&&o[2]==f[8];append!(got,Float64.(o[1]));append!(got,Float64.(o[2]))
  end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":0.0,\"timing_region\":\"two_complete_KNN_hostinclusive_alloc_H2D_3kernels_D2H_free\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),err,rep,path,length(art));ok||error("fullKNN distances and stable exactindices contract")
 end
end
