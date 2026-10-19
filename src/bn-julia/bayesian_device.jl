# Pure scalar/index computation shared by own framework GPU dispatches.
# Source constants are named9node/600sample/state2 study scope.
using StaticArrays
const BN_N=Int32(9)
const BN_D=Int32(600)
@inline function bn_choose(n::Int32,a::Int32)
 result=Int32(1);count=a
 for i in Int32(0):count-Int32(1)
  result*=n;n-=Int32(1)
 end
 for i in Int32(0):count-Int32(1)
  result=div(result,a);a-=Int32(1)
 end
 return result
end
@inline function bn_combination!(comb,l::Int32,n::Int32)
 if l==0
  for i in 1:4;comb[i]=Int32(-1);end
  return nothing
 end
 total=Int32(0);k=Int32(1)
 while total<l
  total+=bn_choose(n,k);k+=Int32(1)
 end
 k-=Int32(1);l-=total-bn_choose(n,k);low=Int32(0);pos=Int32(0)
 while k>1
  total=Int32(0);s=Int32(1)
  while total<l
   total+=bn_choose(n-s,k-Int32(1));s+=Int32(1)
  end
  s-=Int32(1);k-=Int32(1);l-=total-bn_choose(n-s,k);low+=s;comb[pos+1]=low;pos+=Int32(1);n-=s
 end
 comb[pos+1]=low+l
 for i in pos+Int32(1):Int32(3);comb[i+1]=Int32(-1);end
 return nothing
end
@inline function bn_state!(number::Int32,state,time::Int32)
 maximum=Int32(1)
 for i in Int32(0):number-Int32(1);maximum*=Int32(2);end
 time>maximum-Int32(1)&&return false
 if time>=1
  state[1]+=Int32(1)
  if state[1]>=2
   state[1]=Int32(0);n=Int32(1)
   while n<=BN_N
    state[n+1]+=Int32(1)
    if state[n+1]>=2;state[n+1]=Int32(0);n+=Int32(1);else;break;end
   end
  end
 end
 return true
end
@inline function bn_index(parent,size::Int32)
 size==0&&return Int32(0)
 index=Int32(0)
 for i in Int32(1):size-Int32(1);index+=bn_choose(BN_N-Int32(1),i);end
 for i in Int32(1):size-Int32(1)
  for j in parent[i]+Int32(1):parent[i+1]-Int32(1)
   index+=bn_choose(BN_N-Int32(1)-j,size-i)
  end
 end
 return index+parent[size+1]-parent[size]
end
@inline function bn_gen_score!(id::Int32,sizepernode::Int32,localscore,data,lg)
 @inbounds begin
 if id<sizepernode
  parent=MVector{5,Int32}(0,0,0,0,0);pre=MVector{9,Int32}(ntuple(_->Int32(0),9));state=MVector{5,Int32}(0,0,0,0,0);counts=MVector{2,Int32}(0,0)
  bn_combination!(parent,id,BN_N-Int32(1));number=Int32(0)
  for i in 1:4;number+=parent[i]>0;end
  for node in Int32(0):BN_N-Int32(1)
   j=Int32(1)
   for i in Int32(0):BN_N-Int32(1)
    if i!=node;pre[j+1]=i;j+=Int32(1);end
   end
   for k in Int32(0):number-Int32(1);state[k+1]=Int32(0);end
   index=sizepernode*node+id;time=Int32(0)
   while bn_state!(number,state,time)
    time+=Int32(1);ls=0f0;counts[1]=Int32(0);counts[2]=Int32(0)
    for row in Int32(0):BN_D-Int32(1)
     matches=true
     for p in Int32(0):number-Int32(1)
      if data[row*BN_N+pre[parent[p+1]+1]+1]!=state[p+1];matches=false;break;end
     end
     if matches;counts[data[row*BN_N+node+1]+1]+=Int32(1);end
    end
    total=Int32(1)
    for s in 1:2;ls+=lg[counts[s]+1];total+=counts[s];end
    ls-=lg[total+1];ls+=lg[2];localscore[index+1]+=ls
   end
  end
 end
 end
 return nothing
end
@inline function bn_thread_best(id::Int32,taskperthr::Int32,sizepernode::Int32,localscore,mask,node::Int32,total::Int32)
 @inbounds begin
  pos=Int32(1);pre=MVector{9,Int32}(ntuple(_->Int32(0),9));parent=MVector{5,Int32}(-1,0,0,0,0);bestparent=MVector{4,Int32}(0,0,0,0);best=-999999999999999f0
  for i in Int32(0):BN_N-Int32(1)
   if mask[i+1]==1;pre[pos+1]=i;pos+=Int32(1);end
  end
  iteration=Int32(0)
  while iteration<taskperthr && id*taskperthr+iteration<total
   bn_combination!(parent,id*taskperthr+iteration,pos-Int32(1));number=Int32(0)
   while number<4
    parent[number+1]<0&&break
    p=pre[parent[number+1]+1];parent[number+1]=p>node ? p : p+Int32(1);number+=Int32(1)
   end
   for t in number:-Int32(1):Int32(1);parent[t+1]=parent[t];end
   parent[1]=Int32(0);index=bn_index(parent,number)+sizepernode*node;value=localscore[index+1]
   if value>best
    best=value
    for t in 1:4;bestparent[t]=parent[t+1];end
   end
   iteration+=Int32(1)
  end
 end
 return best,bestparent
end
