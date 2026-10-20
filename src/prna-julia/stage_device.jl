@inline function rna_views(c)
 (s=RNAArray{Int32}(c.seq),bcp=RNAArray{Int32}(c.bcp),v=RNAArray{Float32}(c.v),w5=RNAArray{Float32}(c.w5,Int32(1)),w3=RNAArray{Float32}(c.w3),z=RNAArray{Float32}(c.z),yl=RNAArray{Float32}(c.yl),y=RNAArray{Float32}(c.y),wq=RNAArray{Float32}(c.wq),w=RNAArray{Float32}(c.w),wl=RNAArray{Float32}(c.wl),xl=RNAArray{Float32}(c.xl),x=RNAArray{Float32}(c.x),p=RNAParams(c.params))
end
@inline function rna_scalar_stage(::Val{1},c,n,d,i)
 a=rna_views(c);calc_hairpin_stack_exterior_multibranch_body(i,d,n,a.s,a.bcp,a.v,a.x,a.w5,a.w3,a.p);nothing
end
@inline function rna_scalar_stage(::Val{4},c,n,d,i)
 a=rna_views(c);calc_wl_body(i,d,n,a.s,a.bcp,a.v,a.z,a.wq,a.w,a.wl,a.p);nothing
end
@inline function rna_scalar_stage(::Val{7},c,n,d,i)
 a=rna_views(c);calc_x_body(i,d,n,a.yl,a.y,a.w,a.wl,a.xl,a.x,a.p);nothing
end
@inline function rna_partial(::Val{2},c,n,d,i,tx,ty,lane)
 a=rna_views(c);j=wrap(i+d+Int32(1),n)
 if (is_exterior(i,j)&&i-j<=3)||(is_interior(i,j)&&d<=5)||!can_pair(i,j,n,a.bcp);return RNA_INF,RNA_INF,Int32(0);end
 calc_internal_body(i,d,n,a.s,a.bcp,a.v,a.p,tx,ty),RNA_INF,Int32(2)
end
@inline function rna_partial(::Val{3},c,n,d,i,tx,ty,lane)
 a=rna_views(c);j=wrap(i+d+Int32(1),n)
 if (is_exterior(i,j)&&i-j<=3)||!can_pair(i,j,n,a.bcp);return RNA_INF,RNA_INF,Int32(0);end
 calc_coaxial_body(i,d,n,a.s,a.bcp,a.v,a.y,a.w5,a.w3,a.p,lane),RNA_INF,Int32(2)
end
@inline function rna_partial(::Val{5},c,n,d,i,tx,ty,lane)
 a=rna_views(c);j=wrap(i+d+Int32(1),n)
 if is_exterior(i,j)&&i-j<=3;return RNA_INF,RNA_INF,Int32(0);end
 if is_interior(i,j)&&d<=7;return RNA_INF,RNA_INF,Int32(1);end
 calc_xl_body(i,d,n,a.z,a.yl,a.xl,lane),RNA_INF,Int32(2)
end
@inline function rna_partial(::Val{6},c,n,d,i,tx,ty,lane)
 a=rna_views(c);j=wrap(i+d+Int32(1),n)
 if (is_exterior(i,j)&&i-j<=3)||(is_interior(i,j)&&d<=7);return RNA_INF,RNA_INF,Int32(0);end
 v1,v2=calc_z_body(i,d,n,a.s,a.bcp,a.v,a.z,a.xl,a.wq,a.p,lane)
 v1,v2,Int32(2)
end
@inline function rna_partial(::Val{8},c,n,d,i,tx,ty,lane)
 a=rna_views(c);v1,v2=calc_w5_and_w3_body(d,n,a.w5,a.w3,a.wq,lane)
 v1,v2,Int32(2)
end
@inline function rna_finish(::Val{S},c,n,d,i,a,b,action) where S
 if action==0;return nothing;end
 if S==2||S==3
  j=wrap(i+d+Int32(1),n);k=ind(i,j,n)+Int32(1);c.v[k]=free_energy_sum(c.v[k],a)
 elseif S==5
  c.xl[ind(rem(d,Int32(2)),i,n)+Int32(1)]=a
 elseif S==6
  v=rna_views(c);j=wrap(i+d+Int32(1),n)
  if is_interior(i,j)
   k=upper_triangle_index(i,j)+Int32(1);c.wq[k]=free_energy_sum(c.wq[k],free_energy_sum(a,b))
  end
  wcoax=free_energy_sum(a+Int32(2)*v.p.c,(b+Int32(2)*v.p.b)+Int32(2)*v.p.c)
  k=ind(i,j,n)+Int32(1);c.z[k]=free_energy_sum(c.z[k],wcoax)
  k=ind(rem(d,Int32(2)),i,n)+Int32(1);c.xl[k]=free_energy_sum(c.xl[k],wcoax)
 elseif S==8
  c.w5[d+Int32(3)]=free_energy_sum(c.w5[d+Int32(2)],a)
  c.w3[n-d-Int32(1)]=free_energy_sum(c.w3[n-d],b)
 end
 nothing
end
@inline function rna_source_reduce_cuda(value,tid,buf)
 @inbounds buf[tid+Int32(1)]=value
 CUDA.sync_threads();stride=Int32(64)
 while stride>0
  if tid<stride
   @inbounds buf[tid+Int32(1)]=free_energy_sum(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
  end
  CUDA.sync_threads();stride=div(stride,Int32(2))
 end
 result=tid==0 ? (@inbounds buf[1]) : value
 # Close read before reusing this single framework scratch buffer for second sum.
 CUDA.sync_threads()
 result
end
