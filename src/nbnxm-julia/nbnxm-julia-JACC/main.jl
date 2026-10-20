using CUDA,Libdl,SHA,JACC
include(joinpath(@__DIR__,"..","nb_device.jl"))
@inline function nb_entry!(i,j,k,xq,f,shiftvec,fshift,cj,sci,ex,types,coeff,numtypes,rc2,beta,eps,calcShift)
    ix=Int32(i)-Int32(1);jy=Int32(j)-Int32(1)
    block=Int32(k)-Int32(1)
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
const NB_JACC_SPECS=Dict{Tuple{NTuple{3,Int},NTuple{3,Int}},Any}()
function nb_launch_phase(q,a)
    grid=(1,1,Int(q.gridZ));threads=(8,8,1)
    spec=get!(NB_JACC_SPECS,(threads,grid)) do
        JACC.launch_spec(;threads=threads,blocks=grid,shmem_size=0,sync=false,stream=CUDA.legacy_stream())
    end
    # Installed JACC1.4 3D CUDA launch omits spec.stream. study_common sets
    # the actual task stream to legacy; common C++contextfences also protect
    # all allocation/readback handoffs. No package/globalpreferences changes.
    for repetition in 1:Int(q.repeat)
        JACC.parallel_for(spec,(8,8,Int(q.gridZ)),nb_entry!,a...,q.numTypes,q.cutoffSq,q.beta,q.epsFac,q.calcShift!=0)
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
nb_study()
