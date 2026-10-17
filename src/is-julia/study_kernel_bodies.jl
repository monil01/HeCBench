@inline function rand_step(x,a)
 a1=Float64(Base.unsafe_trunc(Int32,0x1p-23*a));a2=a-0x1p23*a1;x1=Float64(Base.unsafe_trunc(Int32,0x1p-23*x));x2=x-0x1p23*x1;t1=a1*x2+a2*x1;t2=Float64(Base.unsafe_trunc(Int32,0x1p-23*t1));z=t1-0x1p23*t2;t3=0x1p23*z+a2*x2;t4=Float64(Base.unsafe_trunc(Int32,0x1p-46*t3));newx=t3-0x1p46*t4
 return newx,0x1p-46*newx
end
@inline function seed_jump(gid,seed,mult)
 gid==0&&return seed;t1=seed;t2=mult;kk=Int64(4)*Int64(gid)
 while kk>1
  ik=kk÷2
  if 2ik==kk;t2=rand_step(t2,t2)[1];kk=ik;else;t1=rand_step(t1,t2)[1];kk-=1;end
 end
 return rand_step(t1,t2)[1]
end
@inline function stage_body!(::Val{ST},tid,gid,bid,grid,keys,hist,backup,indices,ranks,partial,passed,sums,aux,iteration,seed,mult,s)where ST
 if ST==0
  state=seed_jump(gid,seed,mult);state,x=rand_step(state,mult);state,v=rand_step(state,mult);x+=v;state,v=rand_step(state,mult);x+=v;state,v=rand_step(state,mult);x+=v
  @inbounds keys[gid+1]=Base.unsafe_trunc(Int32,512*x)
 elseif ST==1
  @inbounds keys[iteration+1]=iteration;keys[iteration+25]=Int32(2048)-iteration
  for j in 1:5;@inbounds partial[j]=keys[indices[j]+1];end
 elseif ST==2
  @inbounds hist[gid+1]=0
 elseif ST==3
  @inbounds key=keys[gid+1];CUDA.atomic_add!(pointer(hist,Int(key)+1),Int32(1))
 elseif ST==4
  @inbounds s[tid+1]=0;s[256+tid+1]=hist[bid*Int32(256)+tid+1];position=Int32(257)+tid;offset=Int32(1)
  while offset<256
   block_barrier();@inbounds value=s[position]+s[position-offset];block_barrier();@inbounds s[position]=value;offset<<=1
  end
  @inbounds hist[bid*Int32(256)+tid+1]=s[position];block_barrier()
  if tid==0;@inbounds sums[bid+1]=hist[(bid+Int32(1))*Int32(256)];end
 elseif ST==5
  @inbounds s[tid+1]=0;s[257+tid]=sums[tid+1];position=Int32(257)+tid;offset=Int32(1)
  while offset<256
   block_barrier();@inbounds value=s[position]+s[position-offset];block_barrier();@inbounds s[position]=value;offset<<=1
  end
  block_barrier();@inbounds sums[tid+1]=s[position-1]
 elseif ST==6
  @inbounds hist[gid+1]+=sums[bid+1]
 elseif ST==7
  count=Int32(0)
  for j in 1:5
   @inbounds key=partial[j]
   if 0<key<=65535
    @inbounds rank=hist[key];expected=j<=3 ? ranks[j]+iteration : ranks[j]-iteration
    count+=Int32(rank==expected)
   end
  end
  @inbounds passed[1]+=count
 elseif ST==8
  @inbounds backup[gid+1]=keys[gid+1]
 elseif ST==9
  @inbounds value=backup[gid+1];old=CUDA.atomic_add!(pointer(hist,Int(value)+1),Int32(-1));keys[old]=value
 elseif ST==10
  @inbounds s[tid+1]=gid+1<65536 ? Int32(keys[gid+1]>keys[gid+2]) : Int32(0)
  block_barrier();step=Int32(128)
  while step>0
   if tid<step;@inbounds s[tid+1]+=s[tid+step+1];end
   block_barrier();step>>=1
  end
  if tid==0;@inbounds aux[bid+1]=s[1];end
 end
 return
end
