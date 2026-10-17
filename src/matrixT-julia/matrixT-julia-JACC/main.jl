using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function matrix_0!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    @inbounds out[y*N+x+Int32(1)]=input[y*N+x+Int32(1)]
    nothing
end
function matrix_1!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(16)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[y*N+x+Int32(1)]=tile[ty*Int32(16)+tx+Int32(1)]
    nothing
end
function matrix_2!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[ty*Int32(17)+tx+Int32(1)]
    nothing
end
function matrix_3!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[y*N+x+Int32(1)]=tile[tx*Int32(17)+ty+Int32(1)]
    nothing
end
function matrix_4!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    @inbounds out[x*N+y+Int32(1)]=input[y*N+x+Int32(1)]
    nothing
end
function matrix_5!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(16)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(16)+ty+Int32(1)]
    nothing
end
function matrix_6!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(17)+ty+Int32(1)]
    nothing
end
function matrix_7!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(16)+(tx⊻ty)+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(16)+(ty⊻tx)+Int32(1)]
    nothing
end
function matrix_8!(ix,iy,out,input,N,template)
    xglobal=Int32(ix)-Int32(1);yglobal=Int32(iy)-Int32(1);tx=xglobal%Int32(16);ty=yglobal%Int32(16);bx=xglobal÷Int32(16);by=yglobal÷Int32(16)
    oldbx=bx;bx=(bx+by)%(N÷Int32(16));by=oldbx
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=JACC.shared(template)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    JACC.sync_workgroup()
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(17)+ty+Int32(1)]
    nothing
end
function study_prepare(out,input,N,kind)
    kernels=(matrix_0!,matrix_1!,matrix_2!,matrix_3!,matrix_4!,matrix_5!,matrix_6!,matrix_7!,matrix_8!);k=kernels[kind+1];shared=kind in (0,4) ? 0 : (kind in (2,3,6,8) ? 272 : 256);template=CUDA.zeros(Float32,shared)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=(16,16),blocks=(N÷16,N÷16),shmem_size=shared*4,sync=false)
    ()->JACC.parallel_for(k,spec,(N,N),out,input,Int32(N),template)
end
exit(study_main(study_prepare))
