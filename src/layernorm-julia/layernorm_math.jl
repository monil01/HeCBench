using CUDA
# Scalar/VecElement4 load/store and XOR reduction patterns adapted from this
# study's earlier rmsnorm/study_math.jl; origin hash is recorded in provenance.
@inline function lnload(a,av,k,::Val{1})
 @inbounds (a[k+1],)
end
@inline function lnload(a,av,k,::Val{4})
 @inbounds v=av[(k>>2)+1]
 (v[1].value,v[2].value,v[3].value,v[4].value)
end
@inline function lnstore(a,av,k,v,::Val{1})
 @inbounds a[k+1]=v[1]
 nothing
end
@inline function lnstore(a,av,k,v,::Val{4})
 @inbounds av[(k>>2)+1]=(VecElement(v[1]),VecElement(v[2]),VecElement(v[3]),VecElement(v[4]))
 nothing
end
@inline function lnwarp(v)
 for d in (16,8,4,2,1);v=CUDA.shfl_xor_sync(UInt32(0xffffffff),v,Int32(d))+v;end
 v
end
@inline function lncsload(a,i)
 CUDA.CUDACore.@asmcall("ld.global.cs.f32 \$0, [\$1];", "=f,l", true, Float32, Tuple{Core.LLVMPtr{Float32,1}}, pointer(a,i+1))
end
@inline function lncsstore(a,i,v)
 CUDA.CUDACore.@asmcall("st.global.cs.f32 [\$0], \$1;", "l,f", true, Cvoid, Tuple{Core.LLVMPtr{Float32,1},Float32}, pointer(a,i+1),v)
 nothing
end
@inline function lnblock(v,tid,nt,sh)
 lane=tid&31;warp=tid>>5;v=lnwarp(v)
 if lane==0;@inbounds sh[warp+1]=v;end
 CUDA.sync_threads()
 if warp==0
  v=lane<(nt>>5) ? (@inbounds sh[lane+1]) : 0f0;v=lnwarp(v)
  if lane==0;@inbounds sh[1]=v;end
 end
 CUDA.sync_threads()
 @inbounds sh[1]
end
@inline function ln0!(tid,row,nt,sh,out,ov,mean,rstd,x,xv,w,wv,b,bv,N,C,::Val{U}) where U
 lane=tid&31;warp=tid>>5;start=warp*Int32(32U)+lane*Int32(U);step=nt*Int32(U);s=0f0
 for off in start:step:C-Int32(1)
  v=lnload(x,xv,row*C+off,Val(U))
  for j in 1:U;s+=v[j];end
 end
 m=lnblock(s,tid,nt,sh)/Float32(C)
 # Every warp must read the shared result before the same workspace is reused.
 CUDA.sync_threads()
 if tid==0;lncsstore(mean,row,m);end
 s=0f0
 for off in start:step:C-Int32(1)
  v=lnload(x,xv,row*C+off,Val(U))
  for j in 1:U;d=v[j]-m;s=muladd(d,d,s);end
 end
 r=CUDA.rsqrt(lnblock(s,tid,nt,sh)/Float32(C)+1f-5)
 if tid==0;lncsstore(rstd,row,r);end
 for off in start:step:C-Int32(1)
  v=lnload(x,xv,row*C+off,Val(U));g=lnload(w,wv,off,Val(U));bb=lnload(b,bv,off,Val(U))
  values=let vv=v,gg=g,bet=bb,mm=m,rr=r;ntuple(j->muladd((vv[j]-mm)*rr,gg[j],bet[j]),Val(U));end
  lnstore(out,ov,row*C+off,values,Val(U))
 end
 nothing
end
@inline function lnwarp_site!(::Val{K},tid,bid,nt,out,mean,rstd,x,w,b,N,C) where K
 lane=tid&31;row=bid*(nt>>5)+(tid>>5)
 if row>=N;return;end
 s=0f0;s2=0f0
 for j in lane:Int32(32):C-Int32(1)
  @inbounds v=x[row*C+j+1];s+=v
  if K==2;s2=muladd(v,v,s2);end
 end
 m=lnwarp(s)/Float32(C)
 if K==1
  s=0f0
  for j in lane:Int32(32):C-Int32(1);@inbounds d=x[row*C+j+1]-m;s=muladd(d,d,s);end
  variance=lnwarp(s)/Float32(C)
 else
  second=lnwarp(s2)/Float32(C);variance=muladd(-m,m,second)
 end
 r=CUDA.rsqrt(variance+1f-5)
 if lane==0;lncsstore(mean,row,m);lncsstore(rstd,row,r);end
 for j in lane:Int32(32):C-Int32(1)
  v=lncsload(x,row*C+j);@inbounds y=muladd(r*(v-m),w[j+1],b[j+1]);lncsstore(out,row*C+j,y)
 end
 nothing
end
