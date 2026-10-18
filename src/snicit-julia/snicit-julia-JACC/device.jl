# Explicit source-index and source-arithmetic transcription. Source offsets remain zero based.
@inline clip1(v)=min(1f0,max(0f0,v))
@inline function snicit_body(::Val{OP},Y0,W,B,Y1,RP,COL,V,ROWS,LUT,NE,YS,LABEL,CNT,M,N,K,C) where OP
 x=Int32(CUDA.threadIdx().x-Int32(1));y=Int32(CUDA.threadIdx().y-Int32(1))
 bx=Int32(CUDA.blockDim().x);by=Int32(CUDA.blockDim().y);row=Int32(CUDA.blockIdx().x-Int32(1));tid=x+y*bx
 if OP==Int32(0) || OP==2
  sh=CUDA.CuDynamicSharedArray(Float32,Int(K))
  if y==Int32(0);@inbounds sh[x+Int32(1)]=B[x+Int32(1)];end
  CUDA.sync_threads()
  for i=y:by:N-Int32(1)
   @inbounds v=Y0[row*N+i+Int32(1)]
   if v!=0f0;@inbounds z=v*W[i*K+x+Int32(1)];CUDA.@atomic sh[x+Int32(1)]+=z;end
  end
  CUDA.sync_threads()
  if y==Int32(0);@inbounds Y1[row*K+x+Int32(1)]=OP==Int32(0) ? clip1(sh[x+Int32(1)]) : sh[x+Int32(1)];end
 elseif OP==Int32(1) || OP==6
  rid=OP==6 ? (@inbounds ROWS[row+Int32(1)]) : row
  sh=CUDA.CuDynamicSharedArray(Float32,Int(K))
  if tid<K;@inbounds sh[tid+Int32(1)]=OP==Int32(1) ? B[tid+Int32(1)] : 0f0;end
  CUDA.sync_threads()
  for i=y:by:N-Int32(1)
   @inbounds v=Y0[rid*N+i+Int32(1)]
   if v!=0f0
    @inbounds beg=RP[i+Int32(1)]+x;last=RP[i+2]-Int32(1)
    for j=beg:bx:last
     @inbounds col=COL[j+Int32(1)];z=v*V[j+Int32(1)]
     CUDA.@atomic sh[col+Int32(1)]+=z
    end
   end
  end
  CUDA.sync_threads()
  if tid<K;@inbounds Y1[rid*K+tid+Int32(1)]=OP==Int32(1) ? clip1(sh[tid+Int32(1)]) : sh[tid+Int32(1)];end
 elseif OP==3
  sh=CUDA.CuDynamicSharedArray(Int32,Int32(1))
  if x==Int32(0);sh[Int32(1)]=Int32(0);end
  CUDA.sync_threads()
  for r=x:bx:M-Int32(1)
   best=Int32(Int32(0));v=-10000f0
   for j=Int32(Int32(0)):K-Int32(1)
    @inbounds z=Y0[r*K+j+Int32(1)]
    if z>v;v=z;best=j;end
   end
   if best==(@inbounds LABEL[r+Int32(1)]);CUDA.@atomic sh[Int32(1)]+=Int32(Int32(1));end
  end
  CUDA.sync_threads()
  if x==Int32(0);@inbounds CNT[Int32(1)]=sh[Int32(1)];end
 elseif OP==4
  r=y*M÷by;sh=CUDA.CuDynamicSharedArray(Float32,Int(N+Int32(256)))
  if x==Int32(0);@inbounds sh[N+Int32(129)+y]=Float32(r);end
  CUDA.sync_threads()
  for i=Int32(Int32(0)):Int32(Int32(127))
   if (@inbounds sh[N+Int32(129)+i]) != -1f0
    if tid<N;@inbounds sh[tid+Int32(1)]=Y0[N*Int32(sh[N+Int32(129)+i])+tid+Int32(1)];end
    if tid<Int32(128);@inbounds sh[N+tid+Int32(1)]=0f0;end
    CUDA.sync_threads()
    if (@inbounds sh[N+Int32(129)+y]) != -1f0
     for j=x:bx:N-Int32(1)
      if abs((@inbounds Y0[N*r+j+Int32(1)]-sh[j+Int32(1)]))>0.03f0;CUDA.@atomic sh[N+y+Int32(1)]+=1f0;end
     end
    end
    CUDA.sync_threads()
    if y!=i && (@inbounds sh[N+y+Int32(1)])<Float32(N)*0.03f0;@inbounds sh[N+Int32(129)+y]=-1f0;end
    CUDA.sync_threads()
   end
  end
  if tid<Int32(128);@inbounds YS[tid+Int32(1)]=Int32(sh[N+Int32(129)+tid]);end
  CUDA.sync_threads()
 elseif OP==5
  if (@inbounds LUT[row+Int32(1)]) == -Int32(1)
   @inbounds NE[row+Int32(1)]=true
  else
   sh=CUDA.CuDynamicSharedArray(Float32,Int(N+C))
   if tid<N;@inbounds sh[tid+Int32(1)]=Y0[row*N+tid+Int32(1)];end
   if tid<C;@inbounds sh[N+tid+Int32(1)]=0f0;end
   CUDA.sync_threads()
   for j=x:bx:N-Int32(1)
    if abs((@inbounds Y0[N*YS[y+Int32(1)]+j+Int32(1)]-sh[j+Int32(1)]))>0.04f0;CUDA.@atomic sh[N+y+Int32(1)]+=1f0;end
   end
   CUDA.sync_threads()
   if tid==Int32(0)
    v=Float32(N+Int32(1));best=Int32(-Int32(10))
    for j=Int32(Int32(0)):C-Int32(1)
     if v>(@inbounds sh[N+j+Int32(1)]);@inbounds v=sh[N+j+Int32(1)];best=YS[j+Int32(1)];end
    end
    @inbounds LUT[row+Int32(1)]=best
   end
   CUDA.sync_threads()
   best=@inbounds LUT[row+Int32(1)];v=0f0
   if tid<N
    @inbounds z=sh[tid+Int32(1)]-Y0[N*best+tid+Int32(1)]
    v=abs(z)>0.04f0 ? z : 0f0
    @inbounds Y0[row*N+tid+Int32(1)]=v
   end
   n=CUDA.sync_threads_count(v>0f0)
   if tid==Int32(0);@inbounds NE[row+Int32(1)]=n>Int32(0);end
  end
 elseif OP==7
  rid=@inbounds ROWS[row+Int32(1)];b=@inbounds B[x+Int32(1)]
  if (@inbounds LUT[rid+Int32(1)]) == -Int32(1)
   @inbounds Y1[rid*N+x+Int32(1)]=clip1(Y0[rid*N+x+Int32(1)]+b);NE[rid+Int32(1)]=true
  else
   @inbounds wy=Y0[N*LUT[rid+Int32(1)]+x+Int32(1)];delta=Y0[N*rid+x+Int32(1)]
   td=clip1((wy+b)+delta)-clip1(wy+b)
   v=Float64(abs(td))>0.05 ? td : 0f0
   n=CUDA.sync_threads_count(v!=0f0)
   @inbounds Y1[rid*N+x+Int32(1)]=v
   if x==Int32(0);@inbounds NE[rid+Int32(1)]=n>Int32(0);end
  end
 elseif OP==8
  if (@inbounds LUT[row+Int32(1)]) != -Int32(1)
   sh=CUDA.CuDynamicSharedArray(Float32,Int(N))
   @inbounds sh[x+Int32(1)]=Y0[row*N+x+Int32(1)]+Y0[LUT[row+Int32(1)]*N+x+Int32(1)]
   CUDA.sync_threads()
   @inbounds Y0[row*N+x+Int32(1)]=sh[x+Int32(1)]
  end
 end
 nothing
end
