using KernelAbstractions
include(joinpath(@__DIR__,"..","thermodynamic_device.jl"))
include(joinpath(@__DIR__,"..","stage_device.jl"))
@kernel function init_kernel(c,n)
    i=Int32(@index(Group,Linear))-Int32(1)
    @inbounds c.w5[i+Int32(1)]=0
    @inbounds c.w3[i+Int32(1)]=0
end
@kernel function hairpin_kernel(c,n,d)
    i=Int32(@index(Group,Linear))-Int32(1)
    @inbounds rna_scalar_stage(Val(1),c,n,d,i)
end
@kernel function exterior_kernel(c,n,d)
    i=Int32(@index(Group,Linear))-Int32(1)
    @inbounds rna_scalar_stage(Val(3),c,n,d,i)
end
@kernel function multibranch_kernel(c,n,d)
    i=Int32(@index(Group,Linear))-Int32(1)
    @inbounds rna_scalar_stage(Val(4),c,n,d,i)
end
@kernel function W_kernel(c,n,d)
    i=Int32(@index(Group,Linear))-Int32(1)
    @inbounds rna_scalar_stage(Val(6),c,n,d,i)
end
@kernel function bulge_kernel(c,n,d)
    lane=Int32(@index(Local,Linear))-Int32(1)
    cell=Int32(@index(Group,Linear))-Int32(1)
    tx=rem(lane,Int32(16))
    ty=div(lane,Int32(16))
    tid=tx*Int32(16)+ty
    a,b,active=rna_partial(Val(2),c,n,d,cell,tx,ty,lane)
    buf=@localmem Int32 (256,)
    # The activity flag depends only on the group cell and stage, not the lane.
    if active
        @inbounds buf[tid+Int32(1)]=a
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;a=@inbounds buf[1];end
        @synchronize
    end
    if tid==0;@inbounds rna_finish(Val(2),c,n,d,cell,a,b,active);end
end
@kernel function coaxial_kernel(c,n,d)
    lane=Int32(@index(Local,Linear))-Int32(1)
    cell=Int32(@index(Group,Linear))-Int32(1)
    tx=lane
    ty=Int32(0)
    tid=lane
    a,b,active=rna_partial(Val(5),c,n,d,cell,tx,ty,lane)
    buf=@localmem Int32 (256,)
    # The activity flag depends only on the group cell and stage, not the lane.
    if active
        @inbounds buf[tid+Int32(1)]=a
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;a=@inbounds buf[1];end
        @synchronize
    end
    if tid==0;@inbounds rna_finish(Val(5),c,n,d,cell,a,b,active);end
end
@kernel function WM_kernel(c,n,d)
    lane=Int32(@index(Local,Linear))-Int32(1)
    cell=Int32(@index(Group,Linear))-Int32(1)
    tx=lane
    ty=Int32(0)
    tid=lane
    a,b,active=rna_partial(Val(7),c,n,d,cell,tx,ty,lane)
    buf=@localmem Int32 (256,)
    # The activity flag depends only on the group cell and stage, not the lane.
    if active
        @inbounds buf[tid+Int32(1)]=a
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;a=@inbounds buf[1];end
        @synchronize
    end
    if tid==0;@inbounds rna_finish(Val(7),c,n,d,cell,a,b,active);end
end
@kernel function wl_coax_kernel(c,n,d)
    lane=Int32(@index(Local,Linear))-Int32(1)
    cell=Int32(@index(Group,Linear))-Int32(1)
    tx=lane
    ty=Int32(0)
    tid=lane
    a,b,active=rna_partial(Val(8),c,n,d,cell,tx,ty,lane)
    buf=@localmem Int32 (256,)
    # The activity flag depends only on the group cell and stage, not the lane.
    if active
        @inbounds buf[tid+Int32(1)]=a
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;a=@inbounds buf[1];end
        @synchronize
        @inbounds buf[tid+Int32(1)]=b
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;b=@inbounds buf[1];end
        @synchronize
    end
    if tid==0;@inbounds rna_finish(Val(8),c,n,d,cell,a,b,active);end
end
@kernel function w5w3_kernel(c,n,d)
    lane=Int32(@index(Local,Linear))-Int32(1)
    cell=Int32(0)
    tx=lane
    ty=Int32(0)
    tid=lane
    a,b,active=rna_partial(Val(9),c,n,d,cell,tx,ty,lane)
    buf=@localmem Int32 (256,)
    # The activity flag depends only on the group cell and stage, not the lane.
    if active
        @inbounds buf[tid+Int32(1)]=a
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;a=@inbounds buf[1];end
        @synchronize
        @inbounds buf[tid+Int32(1)]=b
        @synchronize
        stride=Int32(128)
        while stride>0
            if tid<stride
                @inbounds buf[tid+Int32(1)]=min(buf[tid+Int32(1)],buf[tid+stride+Int32(1)])
            end
            @synchronize
            stride=div(stride,Int32(2))
        end
        if tid==0;b=@inbounds buf[1];end
        @synchronize
    end
    if tid==0;@inbounds rna_finish(Val(9),c,n,d,cell,a,b,active);end
end
launch_init(c,n)=init_kernel(KernelAbstractions.get_backend(c.v),1)(c,n;ndrange=Int(n)+1)
function launch_step(c,n,d)
    backend=KernelAbstractions.get_backend(c.v)
    hairpin_kernel(backend,1)(c,n,d;ndrange=1*Int(n))
    bulge_kernel(backend,256)(c,n,d;ndrange=256*Int(n))
    exterior_kernel(backend,1)(c,n,d;ndrange=1*Int(n))
    multibranch_kernel(backend,1)(c,n,d;ndrange=1*Int(n))
    coaxial_kernel(backend,256)(c,n,d;ndrange=256*Int(n))
    W_kernel(backend,1)(c,n,d;ndrange=1*Int(n))
    WM_kernel(backend,256)(c,n,d;ndrange=256*Int(n))
    wl_coax_kernel(backend,256)(c,n,d;ndrange=256*Int(n))
    w5w3_kernel(backend,256)(c,n,d;ndrange=256*1)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rna_study()
