# Complete per-cell and per-block work mappings for the ten original GPU stages.
@inline function rna_views(c)
    (s=RNAArray{Int32}(c.seq),v=RNAArray{Int32}(c.v),w=RNAArray{Int32}(c.w),
     wm=RNAArray{Int32}(c.wm),wca=RNAArray{Int32}(c.wca),
     w5=RNAArray{Int32}(c.w5,Int32(1)),w3=RNAArray{Int32}(c.w3),p=RNAParams(c.params))
end
@inline function rna_scalar_stage(::Val{1},c,n,d,i)
    x=rna_views(c);calc_V_hairpin_and_V_stack_body(i,d,n,x.s,x.v,x.p);nothing
end
@inline function rna_scalar_stage(::Val{3},c,n,d,i)
    x=rna_views(c);calc_V_exterior_body(i,d,n,x.s,x.v,x.w5,x.w3,x.p);nothing
end
@inline function rna_scalar_stage(::Val{4},c,n,d,i)
    x=rna_views(c);calc_V_multibranch_body(i,d,n,x.s,x.v,x.wm,x.p);nothing
end
@inline function rna_scalar_stage(::Val{6},c,n,d,i)
    x=rna_views(c);calc_W_body(i,d,n,x.s,x.v,x.w,x.p);nothing
end
@inline function rna_partial(::Val{2},c,n,d,i,tx,ty,lane)
    x=rna_views(c);j=wrap(i+d+Int32(1),n)
    if (is_exterior(i,j)&&i-j<=3)||(is_interior(i,j)&&d<=5)||!can_pair(i,j,n,x.s)
        return RNA_INF,RNA_INF,false
    end
    calc_V_bulge_internal_body(i,d,n,x.s,x.v,x.p,tx,ty),RNA_INF,true
end
@inline function rna_partial(::Val{5},c,n,d,i,tx,ty,lane)
    x=rna_views(c);j=wrap(i+d+Int32(1),n)
    if (is_exterior(i,j)&&i-j<=3)||!can_pair(i,j,n,x.s)
        return RNA_INF,RNA_INF,false
    end
    calc_coaxial_body(i,d,n,x.s,x.v,x.w,x.w5,x.w3,x.p,lane),RNA_INF,true
end
@inline function rna_partial(::Val{7},c,n,d,i,tx,ty,lane)
    x=rna_views(c);value,active=calc_WM_body(i,d,n,x.s,x.w,x.wm,x.p,lane)
    value,RNA_INF,active
end
@inline function rna_partial(::Val{8},c,n,d,i,tx,ty,lane)
    x=rna_views(c);calc_wl_coax_body(i,d,n,x.s,x.v,x.w,x.wm,x.p,x.wca,lane)
end
@inline function rna_partial(::Val{9},c,n,d,i,tx,ty,lane)
    x=rna_views(c);a,b=calc_w5_and_w3_body(d,n,x.s,x.v,x.w5,x.w3,x.p,x.wca,lane)
    a,b,true
end
@inline function rna_finish(::Val{2},c,n,d,i,a,b,active)
    if active
        j=wrap(i+d+Int32(1),n);index=ind(i,j,n)+Int32(1)
        c.v[index]=min(c.v[index],a)
    end
    nothing
end
@inline rna_finish(::Val{5},c,n,d,i,a,b,active)=rna_finish(Val(2),c,n,d,i,a,b,active)
@inline function rna_finish(::Val{7},c,n,d,i,a,b,active)
    j=wrap(i+d+Int32(1),n);index=ind(i,j,n)+Int32(1)
    c.wm[index]=active ? a : RNA_INF
    if active;c.w[index]=min(c.w[index],a);end
    nothing
end
@inline function rna_finish(::Val{8},c,n,d,i,a,b,active)
    if active
        x=rna_views(c);j=wrap(i+d+Int32(1),n);index=ind(i,j,n)+Int32(1)
        c.wca[index]=min(a,b)
        c.wm[index]=min(c.wm[index],a+Int32(2)*x.p.c,b+Int32(2)*x.p.b+Int32(2)*x.p.c)
        c.w[index]=min(c.w[index],c.wm[index])
    end
    nothing
end
@inline function rna_finish(::Val{9},c,n,d,i,a,b,active)
    # W5 device allocation includes logical element -1 at Julia index one.
    c.w5[d+Int32(3)]=min(c.w5[d+Int32(2)],a)
    c.w3[n-d-Int32(1)]=min(c.w3[n-d],b)
    nothing
end
@inline function rna_cuda_min(value,tid,buf)
    @inbounds buf[tid+Int32(1)]=value
    CUDA.sync_threads()
    stride=Int32(128)
    while stride>0
        if tid<stride
            @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
        end
        CUDA.sync_threads();stride=div(stride,Int32(2))
    end
    # Only native thread zero reads the reduction result; other private values
    # remain unchanged. Explicit closure fence makes buffer reuse well defined.
    result=tid==0 ? (@inbounds buf[1]) : value
    CUDA.sync_threads()
    result
end
@inline function rna_reduce_cuda(stage::Val{S},c,n,d,cell,tx,ty,lane,tid) where S
    a,b,active=rna_partial(stage,c,n,d,cell,tx,ty,lane)
    buf=CUDA.CuStaticSharedArray(Int32,256)
    if active
        a=rna_cuda_min(a,tid,buf)
        if S==8||S==9;b=rna_cuda_min(b,tid,buf);end
    end
    if tid==0;rna_finish(stage,c,n,d,cell,a,b,active);end
    nothing
end
