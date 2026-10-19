using Libdl,Printf
function readcases()
 open(joinpath(@__DIR__,"fixtures.bin")) do f
  cs=[]
  for _ in 1:read(f,UInt32)
   h=read!(f,Vector{UInt32}(undef,6));v=read!(f,Vector{Float32}(undef,2));gene=read!(f,Vector{Float32}(undef,h[5]));expected=read!(f,Vector{Float64}(undef,h[6]))
   push!(cs,(n=Int32(h[1]),D=Int32(h[2]),m=Int32(h[3]),repeat=Int(h[4]),thr=v[1],overlap=v[2],geneinput=gene,expected=expected))
  end
  cs
 end
end
function guarded(::Type{T},n,sentinel)where T
 host=fill(T(sentinel),n+16);host[9:end-8].=T(0);parent=CuArray(host);(parent=parent,data=view(parent,9:n+8))
end
function setup(c)
 (;c...,gene=guarded(Float32,length(c.geneinput),-777),sample=guarded(UInt8,c.m*c.D,165),data=guarded(UInt8,c.m*c.n,165),score=guarded(Float32,c.m,-777),dc=guarded(Int32,c.m,-777),sc=guarded(Int32,c.m,-777),tmps=guarded(UInt8,c.m*c.D,165),tmpd=guarded(UInt8,c.m*c.n,165))
end
function hostcor(gene,D,mask,wid,k,i)
 sx=0f0;sn=0f0;sy=0f0;syn=0f0;sxx=0f0;sxxn=0f0;sxy=0f0;sxyn=0f0;syy=0f0;syyn=0f0
 @inbounds for j in 1:D
  a=gene[k*(D+1)+j];if mask[j];sx+=a;else;sn+=a;end
 end
 sx/=Float32(wid);sn/=Float32(D-wid)
 @inbounds for j in 1:D
  a=gene[k*(D+1)+j];if mask[j];sxx+=(sx-a)*(sx-a);else;sxxn+=(sn-a)*(sn-a);end
 end
 sxx=Float32(sqrt(Float64(sxx)));sxxn=Float32(sqrt(Float64(sxxn)))
 @inbounds for j in 1:D
  b=gene[i*(D+1)+j];if mask[j];sy+=b;else;syn+=b;end
 end
 sy/=Float32(wid);syn/=Float32(D-wid)
 @inbounds for j in 1:D
  a=gene[k*(D+1)+j];b=gene[i*(D+1)+j]
  if mask[j];sxy+=(sx-a)*(sy-b);syy+=(sy-b)*(sy-b)
  else;sxyn+=(sn-a)*(syn-b);syyn+=(syn-b)*(syn-b);end
 end
 abs(sxy/(sxx*Float32(sqrt(Float64(syy))))),abs(sxyn/(sxxn*Float32(sqrt(Float64(syyn)))))
end
function mergeprint!(c,sample,data,score,dc,sc)
 n,D,m=Int(c.n),Int(c.D),Int(c.m)
 for k in 0:m-1
  score[k+1]>=.01&&continue
  for kk in k+1:m-1
   score[kk+1]>=.01&&continue
   common=0;un=0
   for i in 1:n
    a=data[k*n+i]==49;b=data[kk*n+i]==49;common+=a&&b;un+=a||b
   end
   observed=Float32(common)/Float32(un)*100f0
   if observed>=c.overlap&&observed<100f0
    mask=[sample[k*D+j]==49||sample[kk*D+j]==49 for j in 1:D];wid=sum(mask);g=[data[k*n+i]==49||data[kk*n+i]==49 for i in 1:n];tot=mn=dif=uni=0
    for i in 0:n-1
     g[i+1]||continue
     for l in i+1:n-1
      g[l+1]||continue
      r,nr=hostcor(c.geneinput,D,mask,wid,i,l);a=r>c.thr;b=nr>c.thr;tot+=a;mn+=b;dif+=a&&b;uni+=a||b
     end
    end
    s=uni>0&&tot>mn ? Float32(dif)/Float32(uni) : 1f0
   elseif observed==100f0;s=score[k+1]
   else;s=1f0;end
   if s<.01f0
    for j in 1:n
     if data[kk*n+j]==49&&data[k*n+j]==48;data[k*n+j]=49;dc[k+1]+=1;end
    end
    for j in 1:D
     if sample[kk*D+j]==49&&sample[k*D+j]==48;sample[k*D+j]=49;sc[k+1]+=1;end
    end
    score[kk+1]=1f0;score[k+1]=s
   end
  end
 end
 io=IOBuffer()
 for k in 0:m-1
  if score[k+1]<.01&&dc[k+1]>10&&sc[k+1]>10
   for i in 0:n-1;if data[k*n+i+1]==49;print(io,i," ");end;end;print(io,'\n')
   for j in 0:D-1;if sample[k*D+j+1]==49;print(io,j," ");end;end;print(io,"\n\n")
  end
 end
 take!(io)
end
function runstudy(cs)
 prefix=Float64[];out=Float64[];sec=0.
 for raw in cs
  c=setup(raw);copyto!(c.gene.data,c.geneinput)
  for b in(c.gene,c.sample,c.data,c.score,c.dc,c.sc,c.tmps,c.tmpd);append!(prefix,Float64.(Array(b.data)));end
  CUDA.device_synchronize();start=time_ns()
  for _ in 1:c.repeat;launch(c);end
  CUDA.device_synchronize();sec+=(time_ns()-start)*1e-9/c.repeat
  sample=Array(c.sample.data);data=Array(c.data.data);score=Array(c.score.data);dc=Array(c.dc.data);sc=Array(c.sc.data)
  for v in(sample,data,score,dc,sc,Array(c.tmps.data),Array(c.tmpd.data));append!(out,Float64.(v));end
  text=mergeprint!(c,sample,data,score,dc,sc)
  for v in(sample,data,score,dc,sc);append!(out,Float64.(v));end
  push!(out,length(text));append!(out,Float64.(text));Array(c.gene.data)==c.geneinput||error("readonlygene")
  for b in(c.gene,c.score,c.dc,c.sc,c.sample,c.data,c.tmps,c.tmpd)
   h=Array(b.parent);append!(out,Float64.(h[1:8]));append!(out,Float64.(h[end-7:end]))
  end
 end
 append!(prefix,out);sec,prefix
end
function main()
 cs=readcases();checker=Libdl.dlsym(Libdl.dlopen(joinpath(@__DIR__,"study_check.so")),:study_check);runstudy(cs)
 for trial in 1:5
  sec,v=runstudy(cs);path="trial$(trial).bin";write(path,v);met=zeros(Cdouble,2)
  ccall(checker,Cint,(Cstring,Ptr{Cdouble}),path,met)==0||error("independentCCSoracleFAIL")
  @printf("\nSTUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"source_bicluster_kernel_repeat_mean_sum3cases\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,length(v),path,length(v))
 end
end
