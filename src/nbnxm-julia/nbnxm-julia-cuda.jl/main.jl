using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"..","nb_device.jl"))
function nb_entry!(xq,f,shiftvec,fshift,cj,sci,ex,types,coeff,numtypes,rc2,beta,eps,calcShift)
    ix=Int32(threadIdx().x)-Int32(1);jy=Int32(threadIdx().y)-Int32(1)
    block=Int32(blockIdx().z)-Int32(1)
    @inbounds sc=sci[Int32(4)*block+Int32(1)]
    @inbounds shift=sci[Int32(4)*block+Int32(2)]
    @inbounds jstart=sci[Int32(4)*block+Int32(3)]
    @inbounds jend=sci[Int32(4)*block+Int32(4)]
    smx=CuStaticSharedArray(Float32,256)
    smt=CuStaticSharedArray(Int32,64)
    red=CuStaticSharedArray(Float32,192)
    nb_preload!(ix,jy,sc,shift,xq,shiftvec,types,smx,smt,eps)
    sync_threads()
    fx,fy,fz=nb_pairs!(ix,jy,sc,shift,jstart,jend,xq,f,cj,ex,types,coeff,smx,smt,numtypes,rc2,beta)
    tid=jy*Int32(8)+ix;shiftbuf=0f0;doShift=calcShift&&(shift!=Int32(22))
    Base.Cartesian.@nexprs 8 c -> begin
        @inbounds red[tid+Int32(1)]=fx[c]
        @inbounds red[Int32(64)+tid+Int32(1)]=fy[c]
        @inbounds red[Int32(128)+tid+Int32(1)]=fz[c]
        sync_threads()
        offset=Int32(4)
        while offset>Int32(1)
            if jy<offset
                @inbounds red[tid+Int32(1)]+=red[tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(64)+tid+Int32(1)]+=red[Int32(64)+tid+Int32(8)*offset+Int32(1)]
                @inbounds red[Int32(128)+tid+Int32(1)]+=red[Int32(128)+tid+Int32(8)*offset+Int32(1)]
            end
            offset=offset>>Int32(1);sync_threads()
        end
        if jy<Int32(3)
            @inbounds value=red[jy*Int32(64)+ix+Int32(1)]+red[jy*Int32(64)+Int32(8)+ix+Int32(1)]
            atom=(sc*Int32(8)+Int32(c-1))*Int32(8)+ix
            nb_atomic!(f,Int32(3)*atom+jy+Int32(1),value)
            if doShift;shiftbuf+=value;end
        end
        sync_threads()
    end
    if doShift && jy<Int32(3)
        nb_atomic!(fshift,Int32(3)*shift+jy+Int32(1),shiftbuf)
    end
    nothing
end
function nb_launch_phase(q,a)
    for repetition in 1:Int(q.repeat)
        CUDA.@cuda threads=(8,8,1) blocks=(1,1,Int(q.gridZ)) stream=CUDA.legacy_stream() nb_entry!(a...,q.numTypes,q.cutoffSq,q.beta,q.epsFac,q.calcShift!=0)
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
nb_study()
