using StaticArrays
const Z=Int32(0);const O=Int32(1)
@inline function base_body!(reads,total,idx)
 i=Int64(idx)
 @inbounds while i<total
  b=reads[i+1]
  reads[i+1]=b==65||b==97 ? UInt8(0) : b==67||b==99 ? UInt8(1) : b==71||b==103 ? UInt8(2) : b==84||b==116||b==85||b==117 ? UInt8(3) : UInt8(4)
  i+=Int64(16384)
 end
 nothing
end
@inline function compress_body!(lengths,offsets,reads,compressed,gaps,n,idx)
 @inbounds if idx<n
  mark=offsets[idx+O]÷16;round=Z;gap=Z;value=UInt32(0);start=offsets[idx+O];stop=start+lengths[idx+O]
  for i in start:stop-1
   b=reads[i+1]
   if b<4
    value+=UInt32(b)<<(Int32(15)-round)*Int32(2);round+=O
    if round==16;compressed[mark+1]=value;value=UInt32(0);round=Z;mark+=1;end
   else;gap+=O;end
  end
  compressed[mark+1]=value;gaps[idx+O]=gap
 end
 nothing
end
@inline function index_body!(reads,lengths,offsets,index,orders,words,magic,n,idx,::Val{K})where K
 @inbounds if idx<n
  start=Int32(offsets[idx+O]);stop=start+lengths[idx+O];hist=MVector{K,UInt8}(undef)
  for j in 1:K;hist[j]=4;end
  m0=Z;m1=Z;m2=Z;m3=Z;count=Z
  for i in start:Int32(stop-O)
   for j in 1:K-1;hist[j]=hist[j+1];end
   b=reads[i+O];hist[K]=b
   if b==0;m0+=O;elseif b==1;m1+=O;elseif b==2;m2+=O;elseif b==3;m3+=O;end
   value=UInt16(0);flag=Z
   for j in 1:K
    value+=(UInt16(hist[j])&UInt16(3))<<((K-j)*2);flag+=max(Int32(hist[j])-Int32(3),Z)
   end
   index[i+O]=flag!=0 ? UInt16(65535) : value;count+=Int32(flag==0)
  end
  words[idx+O]=Int64(count);magic[idx*Int32(4)+O]=m0;magic[idx*Int32(4)+Int32(2)]=m1;magic[idx*Int32(4)+Int32(3)]=m2;magic[idx*Int32(4)+Int32(4)]=m3
 end
 nothing
end
@inline function cutoff_body!(threshold,k,lengths,words,cutoff,n,idx)
 @inbounds if idx<n
  len=lengths[idx+O];required=len-k+O;removed=ceil(Int32,Float32(len)*(1f0-threshold)*Float32(k));cutoff[idx+O]=required-removed
 end
 nothing
end
@inline function merge_body!(offsets,index,orders,words,n,idx)
 @inbounds if idx<n
  start=Int32(offsets[idx+O]);stop=start+Int32(words[idx+O]);previous=index[start+O];count=O
  for j in start+O:Int32(stop-O)
   now=index[j+O]
   if now==previous;count+=O;orders[j]=0
   else;previous=now;orders[j]=UInt16(count);count=O;end
  end
  orders[stop]=UInt16(count)
 end
 nothing
end
@inline function representative_body!(cluster,rep,n)
 @inbounds r=rep[1]+O
 @inbounds while r<n
  if cluster[r+O]<0;cluster[r+O]=r;break;end
  r+=O
 end
 @inbounds rep[1]=r
 nothing
end
@inline function table_body!(offsets,index,orders,words,table,rep,idx,::Val{CLEAN})where CLEAN
 @inbounds start=Int32(offsets[rep+O]);@inbounds stop=start+Int32(words[rep+O]);i=idx+start
 @inbounds while i<stop
  order=orders[i+O]
  if order!=0;table[Int32(index[i+O])+O]=CLEAN ? UInt16(0) : order;end
  i+=Int32(16384)
 end
 nothing
end
@inline function magic_body!(threshold,lengths,magic,cluster,rep,n,idx)
 @inbounds if idx<n&&cluster[idx+O]<0
  a=rep*Int32(4);b=idx*Int32(4);common=min(magic[a+O],magic[b+O])+min(magic[a+Int32(2)],magic[b+Int32(2)])+min(magic[a+Int32(3)],magic[b+Int32(3)])+min(magic[a+Int32(4)],magic[b+Int32(4)])
  required=ceil(Int32,Float32(lengths[idx+O])*threshold)
  if common>required;cluster[idx+O]=Int32(-2);end
 end
 nothing
end
@inline function filter_sum!(offsets,index,orders,words,cluster,table,n,result,gid,lid)
 active=false
 @inbounds if gid<n&&cluster[gid+O]==-2
  active=true;result[lid+O]=Z;start=Int32(offsets[gid+O]);stop=start+Int32(words[gid+O]);i=lid+start
  while i<stop
   result[lid+O]+=Int32(min(table[Int32(index[i+O])+O],orders[i+O]));i+=Int32(128)
  end
 end
 active
end
@inline function filter_finish!(cutoff,cluster,result,gid,lid)
 @inbounds if lid==0
  for i in Int32(1):Int32(127);result[1]+=result[i+O];end
  cluster[gid+O]=result[1]>cutoff[gid+O] ? Int32(-3) : Int32(-1)
 end
 nothing
end
@inline function align_pack!(rowin,rowout,prev,now,compressed,qoff,toff,tlen,qlen,t32,q32,ipack,resetcol)
 @inbounds for j in 1:17;prev[j]=Z;now[j]=Z;end
 ti=Z;@inbounds query=compressed[qoff+ipack+O]
 @inbounds for j in Z:Int32(t32-O)
  if resetcol;prev[1]=Int32(rowin[ti+O]);end
  target=compressed[toff+j+O];k=Int32(30)
  while k>=0
   tb=Int32((target>>k)&UInt32(3));m=Z;now[1]=Int32(rowin[ti+Int32(2)])
   for l in Int32(30):Int32(-2):Z
    m+=O;qb=Int32((query>>l)&UInt32(3));v=prev[m]+Int32(qb==tb);now[m+O]=max(v,now[m],prev[m+O])
   end
   ti+=O;rowout[ti+O]=Int16(now[17])
   if ti==tlen
    if ipack==q32-O;return true,now[qlen%Int32(16)+O];end
    break
   end
   k-=Int32(2);tb=Int32((target>>k)&UInt32(3));m=Z;prev[1]=Int32(rowin[ti+Int32(2)])
   for l in Int32(30):Int32(-2):Z
    m+=O;qb=Int32((query>>l)&UInt32(3));v=now[m]+Int32(qb==tb);prev[m+O]=max(v,prev[m],now[m+O])
   end
   ti+=O;rowout[ti+O]=Int16(prev[17])
   if ti==tlen
    if ipack==q32-O;return true,prev[qlen%Int32(16)+O];end
    break
   end
   k-=Int32(2)
  end
 end
 false,Z
end
@inline function align_body!(threshold,lengths,offsets,compressed,gaps,rep,cluster,n,idx)
 @inbounds if idx<n&&cluster[idx+O]==-3
  required=ceil(Int32,Float32(lengths[idx+O])*threshold);tlen=lengths[rep+O]-gaps[rep+O];qlen=lengths[idx+O]-gaps[idx+O];t32=tlen÷Int32(16)+O;q32=qlen÷Int32(16)+O;toff=Int32(offsets[rep+O]÷16);qoff=Int32(offsets[idx+O]÷16)
  rownow=MVector{3000,Int16}(undef);rowprev=MVector{3000,Int16}(undef);colprev=MVector{17,Int32}(undef);colnow=MVector{17,Int32}(undef)
  for j in Int32(1):Int32(3000);rownow[j]=0;rowprev[j]=0;end
  i=Z
  while i<q32
   done,score=align_pack!(rowprev,rownow,colprev,colnow,compressed,qoff,toff,tlen,qlen,t32,q32,i,true)
   if done;cluster[idx+O]=score>=required ? rep : Int32(-1);return nothing;end
   i+=O
   done,score=align_pack!(rownow,rowprev,colprev,colnow,compressed,qoff,toff,tlen,qlen,t32,q32,i,false)
   if done;cluster[idx+O]=score>=required ? rep : Int32(-1);return nothing;end
   i+=O
  end
 end
 nothing
end
