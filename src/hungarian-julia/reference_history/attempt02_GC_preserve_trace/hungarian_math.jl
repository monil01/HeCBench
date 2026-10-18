# Source-faithful concurrent Hungarian phases; CUDA backend extensions are explicit.
const HN=Int32(256)
@inline function vg(a,i)
 CUDA.CUDACore.@asmcall("ld.volatile.global.s32 \$0, [\$1];", "=r,l", true, Int32, Tuple{Core.LLVMPtr{Int32,1}}, pointer(a,i+1))
end
@inline function vs(a,i,x)
 CUDA.CUDACore.@asmcall("st.volatile.global.s32 [\$0], \$1;", "l,r", true, Cvoid, Tuple{Core.LLVMPtr{Int32,1},Int32}, pointer(a,i+1),Int32(x))
end
@inline function sg(a,i)
 CUDA.CUDACore.@asmcall("ld.volatile.shared.s32 \$0, [\$1];", "=r,l", true, Int32, Tuple{Core.LLVMPtr{Int32,3}}, pointer(a,i+1))
end
@inline function ss(a,i,x)
 CUDA.CUDACore.@asmcall("st.volatile.shared.s32 [\$0], \$1;", "l,r", true, Cvoid, Tuple{Core.LLVMPtr{Int32,3},Int32}, pointer(a,i+1),Int32(x))
end
@inline function hphase!(::Val{K},tid,bid,nt,nb,S) where K
 @inbounds begin
  i=bid*nt+tid;n=HN;slack=S.slack;cr=S.cover_row;cc=S.cover_column;rs=S.row_of_star_at_column;cs=S.column_of_star_at_row;prime=S.column_of_prime_at_row;green=S.row_of_green_at_column;flags=S.flags
  if K==0
   cr[i+1]=0;cc[i+1]=0;rs[i+1]=-1;cs[i+1]=-1
  elseif K==1 || K==3
   w=CUDA.CuDynamicSharedArray(Int32,256);tm=typemax(Int32)
   # n_rows_per_block=n_cols_per_block=1 in this named n256 configuration.
   j=tid
   if K==1;tm=min(tm,slack[j*n+bid+1]);else;tm=min(tm,slack[bid*n+j+1]);end
   w[tid+1]=tm;CUDA.sync_threads()
   for d in (Int32(128),Int32(64))
    if tid<d;w[tid+1]=min(w[tid+1],w[tid+d+1]);end
    CUDA.sync_threads()
   end
   if tid<32
    for d in (Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
     w[tid+1]=min(w[tid+1],w[tid+d+1]);CUDA.sync_warp()
    end
   end
   if tid==0
    if K==1;S.min_in_rows[bid+1]=w[1];else;S.min_in_cols[bid+1]=w[1];end
   end
  elseif K==2
   slack[i+1]-=S.min_in_rows[(i&255)+1]
  elseif K==4
   slack[i+1]-=S.min_in_cols[(i>>8)+1]
   if i==0;flags[1]=0;S.zeros_size_b[1]=0;end
  elseif K==5
   if slack[i+1]==0
    CUDA.atomic_add!(pointer(flags,1),Int32(1));j=CUDA.atomic_add!(pointer(S.zeros_size_b,1),Int32(1));S.zeros[j+1]=i
   end
  elseif K==6
   w=CUDA.CuDynamicSharedArray(Int32,3)
   if tid==0;ss(w,1,0);end
   while true
    CUDA.sync_threads();if tid==0;ss(w,0,0);end;CUDA.sync_threads()
    j=tid
    while j<S.zeros_size_b[1]
     z=S.zeros[j+1];r=z&255;c=z>>8
     if cr[r+1]==0 && cc[c+1]==0
      if CUDA.atomic_xchg!(pointer(cr,r+1),Int32(1))==0
       if CUDA.atomic_xchg!(pointer(cc,c+1),Int32(1))==0
        rs[c+1]=r;cs[r+1]=c
       else
        cr[r+1]=0;ss(w,0,1);ss(w,1,1)
       end
      end
     end
     j+=nt
    end
    CUDA.sync_threads();if sg(w,0)==0;break;end
   end
   if sg(w,1)!=0;flags[4]=1;end
  elseif K==7
   cr[i+1]=0;cc[i+1]=0;if i==0;flags[2]=0;end
  elseif K==8
   if rs[i+1]>=0;cc[i+1]=1;CUDA.atomic_add!(pointer(flags,2),Int32(1));end
  elseif K==9
   prime[i+1]=-1;green[i+1]=-1
  elseif K==10
   w=CUDA.CuDynamicSharedArray(Int32,3)
   if tid==0;ss(w,1,0);ss(w,2,0);end
   while true
    CUDA.sync_threads();if tid==0;ss(w,0,0);end;CUDA.sync_threads()
    j=tid
    while j<S.zeros_size_b[1]
     z=S.zeros[j+1];r=z&255;c=z>>8;c1=cs[r+1]
     for rep in 1:10
      if vg(cc,c)==0 && vg(cr,r)==0
       ss(w,0,1);ss(w,2,1);prime[r+1]=c
       if c1>=0
        vs(cr,r,1);CUDA.threadfence();vs(cc,c1,0)
       else;ss(w,1,1)
       end
      end
     end
     j+=nt
    end
    CUDA.sync_threads();if !(sg(w,0)!=0 && sg(w,1)==0);break;end
   end
   if tid==0
    if sg(w,2)!=0;flags[4]=1;end
    if sg(w,1)!=0;flags[3]=1;end
   end
  elseif K==11
   c=prime[i+1]
   if c>=0 && cs[i+1]<0
    green[c+1]=i;r=rs[c+1]
    while r>=0
     c=prime[r+1];green[c+1]=r;r=rs[c+1]
    end
   end
  elseif K==12
   r=green[i+1]
   if r>=0 && rs[i+1]<0
    c2=cs[r+1];cs[r+1]=i;rs[i+1]=r
    while c2>=0
     r=green[c2+1];c0=c2;c2=cs[r+1];cs[r+1]=c0;rs[c0+1]=r
    end
   end
  elseif K==13 || K==14
   w=CUDA.CuDynamicSharedArray(Int32,Int(nt));tm=typemax(Int32)
   if K==13
    j=bid*(nt*2)+tid
    while j<n*n
     r1=j&255;c1=j>>8;j2=j+nt;r2=j2&255;c2=j2>>8
     a=(cr[r1+1]==1||cc[c1+1]==1) ? typemax(Int32) : slack[j+1]
     b=(cr[r2+1]==1||cc[c2+1]==1) ? typemax(Int32) : slack[j2+1]
     tm=min(tm,min(a,b));j+=nt*2*nb
    end
   else
    tm=min(S.d_min_in_mat_vect[tid+1],S.d_min_in_mat_vect[tid+nt+1])
   end
   w[tid+1]=tm;CUDA.sync_threads()
   if K==13
    if tid<128;w[tid+1]=min(w[tid+1],w[tid+129]);end;CUDA.sync_threads()
   end
   if tid<64;w[tid+1]=min(w[tid+1],w[tid+65]);end;CUDA.sync_threads()
   if tid<32
    for d in (Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
     w[tid+1]=min(w[tid+1],w[tid+d+1]);CUDA.sync_warp()
    end
   end
   if tid==0
    if K==13;S.d_min_in_mat_vect[bid+1]=w[1];else;S.d_min_in_mat[1]=w[1];end
   end
  elseif K==15
   r=i&255;c=i>>8;v=S.d_min_in_mat[1]
   if cr[r+1]==1 && cc[c+1]==1;slack[i+1]+=v;end
   if cr[r+1]==0 && cc[c+1]==0;slack[i+1]-=v;end
   if i==0;flags[1]=0;S.zeros_size_b[1]=0;end
  end
 end
 nothing
end
