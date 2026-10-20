using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"device_bodies.jl"))
function mm_kernel0(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,fpw,lfpw)
    mm_body0(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,fpw,lfpw,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_kernel1(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,fpw,lfpw)
    mm_body1(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,fpw,lfpw,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_kernel2(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log)
    mm_body2(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_kernel3(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log)
    mm_body3(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_kernel4(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log)
    mm_body4(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_kernel5(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log)
    mm_body5(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_kernel6(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log)
    mm_body6(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))
    return
end
function mm_launch(q,a)
    op=Int(q.op);0<=op<7||error("UnknownMMop")
    if op==0
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel0(a[1],a[2],a[3],a[4],a[5],a[6],Int64(q.nfibers),a[9],a[10],a[11],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.fpw),Int64(q.logFPW))
    elseif op==1
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel1(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.fpw),Int64(q.logFPW))
    elseif op==2
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel2(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC))
    elseif op==3
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel3(a[1],a[2],a[3],a[4],a[5],a[6],Int64(q.nfibers),a[9],a[10],a[11],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC))
    elseif op==4
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel4(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC))
    elseif op==5
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel5(a[1],a[2],a[3],a[4],a[5],a[6],Int64(q.nfibers),a[9],a[10],a[11],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC))
    elseif op==6
        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel6(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC))
    end
    nothing
end
include(joinpath(@__DIR__,"..","secondary_fp64_atomic","study_common.jl"))
mm_study()
