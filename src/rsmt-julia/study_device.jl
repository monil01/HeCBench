# Direct warp-cooperative source algorithm. Zero-based native IDs remain Int16/Int32.
const MASK=typemax(UInt32)
@inline median3(a,b,c)=max(min(a,b),min(max(a,b),c))
@inline function build_mst!(num,pout,x,y,es,ed,dist,src,dst,mind,lane,warp,::Val{P}) where P
 @inbounds begin
  left=num-Int32(1)
  for j in lane:Int32(32):left-Int32(1)
   dist[j+P*warp+1]=typemax(Int32);dst[j+P*warp+1]=Int16(j+1)
  end
  root=Int32(0)
  for count in Int32(0):num-Int32(2)
   lane==0&&(mind[warp+1]=typemax(Int32));CUDA.sync_warp(MASK)
   for j in lane:Int32(32):left-Int32(1)
    ix=j+P*warp+1;d=Int32(dst[ix]);new=abs(x[pout+root+1]-x[pout+d+1])+abs(y[pout+root+1]-y[pout+d+1]);old=dist[ix]
    if old>new;old=new;dist[ix]=new;src[ix]=Int16(root);end
    CUDA.atomic_min!(pointer(mind,warp+1),old*Int32(512)+j)
   end
   CUDA.sync_warp(MASK);j=rem(mind[warp+1],Int32(512));ix=j+P*warp+1;root=Int32(dst[ix]);left-=Int32(1)
   CUDA.sync_warp(MASK)
   if lane==0
    es[Int32(2)*(pout+count+1)-Int32(1)]=src[ix];ed[Int32(2)*(pout+count+1)]=Int16(root)
    dist[ix]=dist[left+P*warp+1];src[ix]=src[left+P*warp+1];dst[ix]=dst[left+P*warp+1]
   end
  end
 end
 nothing
end
@inline function insert_steiner!(num,pout,x,y,es,ed,dist,adj,cnt,lane,warp,::Val{P}) where P
 @inbounds begin
  top=num
  for i in lane:Int32(32):top-Int32(1);cnt[i+P*warp+1]=Int32(0);end
  CUDA.sync_warp(MASK)
  for e in lane:Int32(32):top-Int32(2)
   dist[e+P*warp+1]=Int32(-1);s=Int32(es[Int32(2)*(pout+e+1)-Int32(1)]);d=Int32(ed[Int32(2)*(pout+e+1)])
   if x[pout+d+1]!=x[pout+s+1]||y[pout+d+1]!=y[pout+s+1]
    ps=CUDA.atomic_add!(pointer(cnt,s+P*warp+1),Int32(1));adj[ps+8*(s+P*warp)+1]=Int16(e)
    pd=CUDA.atomic_add!(pointer(cnt,d+P*warp+1),Int32(1));adj[pd+8*(d+P*warp)+1]=Int16(e)
   end
  end
  CUDA.sync_warp(MASK)
  for s in lane:Int32(32):top-Int32(1)
   degree=cnt[s+P*warp+1]
   if degree>=2
    x0=x[pout+s+1];y0=y[pout+s+1]
    for j in Int32(0):degree-Int32(2)
     e1=Int32(adj[j+8*(s+P*warp)+1]);ss=Int32(es[Int32(2)*(pout+e1+1)-Int32(1)]);d1=s!=ss ? ss : Int32(ed[Int32(2)*(pout+e1+1)]);x1=x[pout+d1+1];y1=y[pout+d1+1]
     for k in j+Int32(1):degree-Int32(1)
      e2=Int32(adj[k+8*(s+P*warp)+1]);ss2=Int32(es[Int32(2)*(pout+e2+1)-Int32(1)]);d2=s!=ss2 ? ss2 : Int32(ed[Int32(2)*(pout+e2+1)]);stx=median3(x0,x1,x[pout+d2+1]);sty=median3(y0,y1,y[pout+d2+1]);rd=abs(stx-x0)+abs(sty-y0)
      if rd>0
       CUDA.atomic_max!(pointer(dist,e1+P*warp+1),rd*Int32(512)+e2);CUDA.atomic_max!(pointer(dist,e2+P*warp+1),rd*Int32(512)+e1)
      end
     end
    end
   end
  end
  CUDA.sync_warp(MASK);updated=false;e1=lane
  while CUDA.vote_any_sync(MASK,e1<top-Int32(2))
   ins=false;stx=Int32(0);sty=Int32(0)
   if e1<top-Int32(2)
    d1=dist[e1+P*warp+1]
    if d1>0
     e2=rem(d1,Int32(512))
     if e2>e1
      d2=dist[e2+P*warp+1]
      if e1==rem(d2,Int32(512))
       a=Int32(es[Int32(2)*(pout+e1+1)-Int32(1)]);b=Int32(ed[Int32(2)*(pout+e1+1)]);c=Int32(es[Int32(2)*(pout+e2+1)-Int32(1)]);x0=x[pout+a+1];y0=y[pout+a+1];x1=x[pout+b+1];y1=y[pout+b+1];x2=x[pout+c+1];y2=y[pout+c+1]
       if (x2==x0&&y2==y0)||(x2==x1&&y2==y1);c=Int32(ed[Int32(2)*(pout+e2+1)]);x2=x[pout+c+1];y2=y[pout+c+1];end
       updated=true;ins=true;stx=median3(x0,x1,x2);sty=median3(y0,y1,y2)
      end
     end
    end
   end
   ballot=CUDA.vote_ballot_sync(MASK,ins);before=(UInt32(1)<<lane)-UInt32(1);position=Int32(count_ones(ballot&before))+num
   if ins;x[pout+position+1]=stx;y[pout+position+1]=sty;end
   num+=Int32(count_ones(ballot));e1+=Int32(32)
  end
  CUDA.sync_warp(MASK)
  return num,CUDA.vote_any_sync(MASK,updated)
 end
end
@inline function process_net!(net,idx,x,y,es,ed,dist,src,dst,mind,adj,cnt,lane,warp,p)
 @inbounds begin
  n=idx[net+2]-idx[net+1];pout=Int32(2)*idx[net+1];num=n
  while true
   build_mst!(num,pout,x,y,es,ed,dist,src,dst,mind,lane,warp,p)
   num,updated=insert_steiner!(num,pout,x,y,es,ed,dist,adj,cnt,lane,warp,p)
   updated||break
  end
 end
 nothing
end
@inline function rsmt_body!(::Val{W},p::Val{P},::Val{LARGE},idx,xin,yin,oi,x,y,es,ed,wl,state,nets) where {W,P,LARGE}
 lane=Int32(rem(CUDA.threadIdx().x-1,32));warp=Int32(div(CUDA.threadIdx().x-1,32))
 dist=CUDA.CuStaticSharedArray(Int32,W*P);src=CUDA.CuStaticSharedArray(Int16,W*P);dst=CUDA.CuStaticSharedArray(Int16,W*P);mind=CUDA.CuStaticSharedArray(Int32,W);adj=CUDA.CuStaticSharedArray(Int16,8*W*P);cnt=CUDA.CuStaticSharedArray(Int32,W*P)
 @inbounds while true
  i=Int32(0);which=LARGE ? Int32(2) : Int32(1)
  if lane==0;i=CUDA.atomic_add!(pointer(state,which),Int32(1));end
  i=CUDA.shfl_sync(MASK,i,1)
  bound=LARGE ? state[3] : nets
  i>=bound&&break
  if LARGE
   net=wl[i+1];process_net!(net,idx,x,y,es,ed,dist,src,dst,mind,adj,cnt,lane,warp,p)
  else
   net=i;pin=idx[net+1];num=idx[net+2]-pin;pout=Int32(2)*pin
   lane==0&&(oi[net+1]=pout)
   for j in lane:Int32(32):num-Int32(1);x[pout+j+1]=xin[pin+j+1];end
   for j in lane:Int32(32):num-Int32(1);y[pout+j+1]=yin[pin+j+1];end
   if num==2
    if lane==0;es[Int32(2)*(pout+1)-Int32(1)]=Int16(0);ed[Int32(2)*(pout+1)]=Int16(1);end
   elseif num==3
    xx=Int32(0);yy=Int32(0)
    if lane<3;es[Int32(2)*(pout+lane+1)-Int32(1)]=Int16(lane);ed[Int32(2)*(pout+lane+1)]=Int16(3);xx=x[pout+lane+1];yy=y[pout+lane+1];end
    x1=CUDA.shfl_sync(MASK,xx,2);y1=CUDA.shfl_sync(MASK,yy,2);x2=CUDA.shfl_sync(MASK,xx,3);y2=CUDA.shfl_sync(MASK,yy,3)
    if lane==0;x[pout+4]=median3(xx,x1,x2);y[pout+4]=median3(yy,y1,y2);end
   elseif num<=32
    process_net!(net,idx,x,y,es,ed,dist,src,dst,mind,adj,cnt,lane,warp,p)
   elseif lane==0
    pos=CUDA.atomic_add!(pointer(state,3),Int32(1));wl[pos+1]=net
   end
  end
 end
 if !LARGE&&CUDA.threadIdx().x==1&&CUDA.blockIdx().x==1;@inbounds oi[nets+1]=Int32(2)*idx[nets+1];end
 nothing
end
