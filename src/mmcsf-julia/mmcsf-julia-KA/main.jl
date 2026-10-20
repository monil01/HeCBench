using CUDA,Libdl,SHA,KernelAbstractions
const MM_BACKEND=CUDA.CUDABackend(always_inline=true)
include(joinpath(@__DIR__,"..","mmcsf-julia-cuda.jl","device_bodies.jl"))
@kernel unsafe_indices=true function mm_kernel0(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,fpw,lfpw,bd)
    mm_body0(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,fpw,lfpw,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
@kernel unsafe_indices=true function mm_kernel1(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,fpw,lfpw,bd)
    mm_body1(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,fpw,lfpw,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
@kernel unsafe_indices=true function mm_kernel2(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,bd)
    mm_body2(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
@kernel unsafe_indices=true function mm_kernel3(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,bd)
    mm_body3(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
@kernel unsafe_indices=true function mm_kernel4(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,bd)
    mm_body4(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
@kernel unsafe_indices=true function mm_kernel5(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,bd)
    mm_body5(v,l,i,p0,p1,i1,nf,u0,u1,u2,mode,r,w,log,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
@kernel unsafe_indices=true function mm_kernel6(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,bd)
    mm_body6(v,l,i,p0,p1,i1,p2,i2,nf,u0,u1,u2,u3,mode,r,w,log,Int64(@index(Local,Linear))-1,Int64(@index(Group,Linear))-1,bd)
    nothing
end
function mm_launch(q,a)
    op=Int(q.op);0<=op<7||error("UnknownMMop")
    if op==0
        mm_kernel0(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],Int64(q.nfibers),a[9],a[10],a[11],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.fpw),Int64(q.logFPW),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    elseif op==1
        mm_kernel1(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.fpw),Int64(q.logFPW),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    elseif op==2
        mm_kernel2(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    elseif op==3
        mm_kernel3(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],Int64(q.nfibers),a[9],a[10],a[11],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    elseif op==4
        mm_kernel4(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    elseif op==5
        mm_kernel5(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],Int64(q.nfibers),a[9],a[10],a[11],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    elseif op==6
        mm_kernel6(MM_BACKEND,Int(q.block))(a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],Int64(q.nfibers),a[9],a[10],a[11],a[12],Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC),Int64(q.block);ndrange=Int(q.block)*Int(q.grid))
    end
    nothing
end
include(joinpath(@__DIR__,"..","secondary_fp64_atomic","study_common.jl"))
mm_study()
