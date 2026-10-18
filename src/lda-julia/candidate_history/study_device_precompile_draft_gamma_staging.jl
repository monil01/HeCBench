# Copyright(c)2021 Jisang Yoon, Apache2.0; modifiedJulia translation2026.
# CUDA.jl publicwarp/atomic/barrier APIs: native frameworkfeatures forCUDAjl,
# explicitCUDAdeviceinterop forKA/JACC; no claim of nonCUDAbackend portability.
@inline function lda_warp_sum(val::Float32)
 mask=CUDA.active_mask();offset=Int32(16)
 while offset>0
  val+=CUDA.shfl_down_sync(mask,val,offset)
  offset=offset÷Int32(2)
 end
 return val
end
@inline function lda_reduce(sh,base::Int32,len::Int32,K::Int32,tid::Int32)
 warp=tid÷Int32(32);lane=tid%Int32(32);v=0f0;i=tid;red=Int32(4)*K
 while i<len
  v+=sh[base+i+1];i+=Int32(256)
 end
 v=lda_warp_sum(v)
 if lane==0;sh[red+warp+1]=v;end
 CUDA.sync_threads()
 v=tid<Int32(8) ? sh[red+lane+1] : 0f0
 if warp==0
  v=lda_warp_sum(v)
  if tid==0;sh[red+1]=v;end
 end
 CUDA.sync_threads()
 return sh[red+1]
end
@inline function lda_digamma(x::Float32)
 result=0f0
 while x<7f0
  result-=1f0/x;x+=1f0
 end
 x-=.5f0;xx=1f0/x;xx2=xx*xx;xx4=xx2*xx2
 result+=log(x)+(1f0/24f0)*xx2-(7f0/960f0)*xx4+(31f0/8064f0)*xx4*xx2-(127f0/30720f0)*xx4*xx4
 return result
end
@inline function lda_body!(cols,ptr,mask,counts,initialize::Bool,NW::Int32,D::Int32,K::Int32,iters::Int32,alpha,beta,gamma,grad,newbeta,trainloss,valloss,locks,sh,BX::Int32,TX::Int32,grid::Int32)
 doc=BX;gradbase=K*BX;lossbase=Int32(2)*K;valbase=Int32(3)*K
 while doc<D
  beg=ptr[doc+1];last=ptr[doc+2];gbase=K*doc
  if initialize
   j=TX
   while j<K;gamma[gbase+j+1]=alpha[j+1]+Float32((last-beg)÷K);j+=Int32(256);end
  end
  CUDA.sync_threads()
  j=TX
  while j<K;sh[valbase+j+1]=0f0;j+=Int32(256);end
  iteration=Int32(0)
  while iteration<iters
   j=TX
   while j<K;sh[j+1]=0f0;j+=Int32(256);end
   CUDA.sync_threads()
   record=beg
   while record<last
    w=cols[record+1];validation=mask[record+1]!=UInt8(0);count=counts[record+1];bbase=w*K
    if !validation || iteration+Int32(1)==iters
     j=TX
     while j<K
      sh[K+j+1]=beta[bbase+j+1]*exp(lda_digamma(gamma[gbase+j+1]));j+=Int32(256)
     end
     CUDA.sync_threads()
     phi_sum=lda_reduce(sh,K,K,K,TX);j=TX
     while j<K
      phi=sh[K+j+1]/phi_sum;sh[K+j+1]=phi
      if validation;sh[valbase+j+1]+=phi*count;else;sh[j+1]+=phi*count;end
      j+=Int32(256)
     end
     CUDA.sync_threads()
    end
    if iteration+Int32(1)==iters
     if !validation
      if TX==0
       while CUDA.atomic_cas!(pointer(locks,Int64(w)+1),Int32(0),Int32(1))!=Int32(0);end
      end
      CUDA.sync_threads();j=TX
      while j<K;newbeta[bbase+j+1]+=sh[K+j+1]*count;j+=Int32(256);end
      CUDA.sync_threads()
      if TX==0;locks[w+1]=Int32(0);end
      CUDA.sync_threads()
     end
     j=TX
     while j<K
      value=log(max(beta[bbase+j+1],1f-6));value-=log(max(sh[K+j+1],1f-6));value*=sh[K+j+1];sh[lossbase+j+1]=value;j+=Int32(256)
     end
     CUDA.sync_threads();loss=lda_reduce(sh,lossbase,K,K,TX)*count
     if TX==0
      if validation;valloss[BX+1]+=loss;else;trainloss[BX+1]+=loss;end
     end
     CUDA.sync_threads()
    end
    CUDA.sync_threads();record+=Int32(1)
   end
   j=TX
   while j<K;gamma[gbase+j+1]=sh[j+1]+alpha[j+1];j+=Int32(256);end
   CUDA.sync_threads();iteration+=Int32(1)
  end
  gamma_sum=lda_reduce(sh,Int32(0),Int32(0),K,TX) # overwritten below by actual globalgamma reduction
  # Source ReduceSum receives a globalgamma vector, so use the same tree via sharedloss staging.
  j=TX
  while j<K;sh[lossbase+j+1]=gamma[gbase+j+1];j+=Int32(256);end
  CUDA.sync_threads();gamma_sum=lda_reduce(sh,lossbase,K,K,TX)
  j=TX
  while j<K
   elog=lda_digamma(gamma[gbase+j+1])-lda_digamma(gamma_sum);grad[gradbase+j+1]+=elog;sh[j+1]*=elog;sh[valbase+j+1]*=elog;j+=Int32(256)
  end
  train_component=lda_reduce(sh,Int32(0),K,K,TX);val_component=lda_reduce(sh,valbase,K,K,TX)
  if TX==0;trainloss[BX+1]+=train_component;valloss[BX+1]+=val_component;end
  CUDA.sync_threads();doc+=grid
 end
 return nothing
end
