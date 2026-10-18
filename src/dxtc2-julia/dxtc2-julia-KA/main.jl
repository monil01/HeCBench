using CUDA,KernelAbstractions,StaticArrays
include(joinpath(@__DIR__,"..","compress_math.jl"))
@kernel unsafe_indices=true function compress_study!(permutations,image,result,template,offset)
 idx=Int32(@index(Local,Linear))-1;bid=Int32(@index(Group,Linear))-1+offset
 w=@localmem Float32 (467,)

 if idx<16
  @inbounds c=image[bid*16+idx+1]
  col=F3(Float32(c&0xff)*(1f0/255f0),Float32((c>>8)&0xff)*(1f0/255f0),Float32((c>>16)&0xff)*(1f0/255f0))
  storecolor!(w,idx,col)
 end
 @synchronize
 if idx==0
  sums=loadcolor(w,0)
  for j in 1:15;sums=sums+loadcolor(w,j);end
  @inbounds w[49]=sums.x;w[50]=sums.y;w[51]=sums.z
 end
 @synchronize
 @inbounds sums=F3(w[49],w[50],w[51])
 if idx<16
  diff=loadcolor(w,idx)-sums*(1f0/16f0);k=52+6idx
  @inbounds w[k]=diff.x*diff.x;w[k+1]=diff.x*diff.y;w[k+2]=diff.x*diff.z;w[k+3]=diff.y*diff.y;w[k+4]=diff.y*diff.z;w[k+5]=diff.z*diff.z
 end
 @synchronize
 for d in (8,4,2,1)
  if idx<d
   for j in 0:5;@inbounds w[52+6idx+j]+=w[52+6(idx+d)+j];end
  end
  @synchronize
 end
 if idx<16
  covariance=MVector{6,Float32}(undef)
  for j in 1:6;@inbounds covariance[j]=w[51+j];end
  axis=first_eigen_vector(covariance);col=loadcolor(w,idx)
  @inbounds w[148+idx]=col.x*axis.x+col.y*axis.y+col.z*axis.z
 end
 @synchronize
 if idx<16
  rank=Int32(0)
  for j in 0:15;@inbounds rank+=Int32(w[148+j]<w[148+idx]);end
  @inbounds w[164+idx]=Float32(rank)
 end
 @synchronize
 for j in 0:14
  if idx<16 && idx>j
   @inbounds if w[164+idx]==w[164+j];w[164+idx]+=1f0;end
  end
  @synchronize
 end
 if idx<16
  localcol=loadcolor(w,idx)
  @inbounds rank=Int32(w[164+idx])
 end
 @synchronize
 if idx<16;storecolor!(w,rank,localcol);end
 @synchronize
 colors=MVector{16,F3}(undef)
 for j in 0:15;colors[j+1]=loadcolor(w,j);end
 besterror=typemax(Float32);beststart=UInt16(0);bestend=UInt16(0);bestperm=UInt32(0)
 for j in 0:15
  pi=idx+64j
  if pi>=992;break;end
  @inbounds perm=permutations[pi+1]
  if pi<160;@inbounds w[308+pi]=reinterpret(Float32,perm);end
  er,a,b=eval_permutation4(colors,perm,sums)
  if er<besterror;besterror=er;beststart=a;bestend=b;bestperm=perm;end
 end
 if beststart<bestend;beststart,bestend=bestend,beststart;bestperm ⊻= 0x55555555;end
 @synchronize
 for j in 0:2
  pi=idx+64j
  if pi>=160;break;end
  @inbounds perm=reinterpret(UInt32,w[308+pi])
  er,a,b=eval_permutation3(colors,perm,sums)
  if er<besterror
   besterror=er;beststart=a;bestend=b;bestperm=perm
   if beststart>bestend;beststart,bestend=bestend,beststart;bestperm ⊻= ((~bestperm)>>1)&0x55555555;end
  end
 end
 @inbounds w[180+idx]=besterror;w[244+idx]=Float32(idx)
 @synchronize
 for d in (32,16,8,4,2,1)
  @inbounds er0=w[180+idx]
  er1=idx+d<64 ? w[180+idx+d] : typemax(Float32)
  ind1=idx+d<64 ? w[244+idx+d] : 0f0
  @synchronize
  if er1<er0;@inbounds w[180+idx]=er1;w[244+idx]=ind1;end
  @synchronize
 end
 @inbounds winner=Int32(w[244])
 @synchronize
 if idx==winner
  if beststart==bestend;bestperm=UInt32(0);end
  indices=UInt32(0)
  for j in 0:15
   @inbounds ref=UInt32(w[164+j])
   indices |= ((bestperm>>(2ref))&0x03) << (2j)
  end
  @inbounds result[2bid+1]=(UInt32(bestend)<<16)|UInt32(beststart);result[2bid+2]=indices
 end
end
function launch_compress(model,p,i,r,t,n,offset)
 compress_study!(CUDABackend(),64)(p,i,r,t,offset;ndrange=n*64)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:ka)
