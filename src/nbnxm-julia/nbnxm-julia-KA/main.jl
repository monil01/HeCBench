using CUDA,Libdl,SHA,KernelAbstractions
include(joinpath(@__DIR__,"..","nb_device.jl"))
const NB_KA_BACKEND=CUDA.CUDABackend(always_inline=true)
@kernel function nb_entry!(xq,f,shiftvec,fshift,cj,sci,ex,types,coeff,numtypes,rc2,beta,eps,calcShift)
    local_index=@index(Local,Cartesian);group_index=@index(Group,Cartesian)
    ix=Int32(local_index[1])-Int32(1);jy=Int32(local_index[2])-Int32(1)
    block=Int32(group_index[3])-Int32(1)
    @inbounds sc=sci[Int32(4)*block+Int32(1)]
    @inbounds shift=sci[Int32(4)*block+Int32(2)]
    @inbounds jstart=sci[Int32(4)*block+Int32(3)]
    @inbounds jend=sci[Int32(4)*block+Int32(4)]
    smx=@localmem Float32 (256,)
    smt=@localmem Int32 (64,)
    red=@localmem Float32 (192,)
    nb_preload!(ix,jy,sc,shift,xq,shiftvec,types,smx,smt,eps)
    @synchronize
    fx,fy,fz=nb_pairs!(ix,jy,sc,shift,jstart,jend,xq,f,cj,ex,types,coeff,smx,smt,numtypes,rc2,beta)
    tid=jy*Int32(8)+ix;shiftbuf=0f0;doShift=calcShift&&(shift!=Int32(22))
    # Source-unrolled cluster 0: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[1]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[1]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[1]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(0))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 1: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[2]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[2]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[2]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(1))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 2: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[3]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[3]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[3]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(2))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 3: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[4]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[4]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[4]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(3))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 4: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[5]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[5]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[5]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(4))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 5: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[6]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[6]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[6]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(5))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 6: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[7]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[7]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[7]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(6))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    # Source-unrolled cluster 7: lexical barriers visible to KA.
        @inbounds red[tid+Int32(1)]=fx[8]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[8]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[8]
        @synchronize
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);@synchronize
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(7))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        @synchronize
    if doShift && jy<Int32(3)
        nb_atomic!(fshift,Int32(3)*shift+jy+Int32(1),shiftbuf)
    end
    nothing
end
function nb_launch_phase(q,a)
    kernel=nb_entry!(NB_KA_BACKEND,(8,8,1))
    for repetition in 1:Int(q.repeat)
        kernel(a...,q.numTypes,q.cutoffSq,q.beta,q.epsFac,q.calcShift!=0;ndrange=(8,8,Int(q.gridZ)))
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
nb_study()
