@inline function bn_ka_thread_best(id::Int32,taskperthr::Int32,sizepernode::Int32,localscore,mask,node::Int32,total::Int32)
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
 return best,(bestparent[1],bestparent[2],bestparent[3],bestparent[4])
end
