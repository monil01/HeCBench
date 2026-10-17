using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function matrix_0!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    @inbounds out[y*N+x+Int32(1)]=input[y*N+x+Int32(1)]
    nothing
end
@kernel function matrix_1!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (256,)
    @inbounds tile[ty*Int32(16)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[y*N+x+Int32(1)]=tile[ty*Int32(16)+tx+Int32(1)]
    nothing
end
@kernel function matrix_2!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (272,)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[ty*Int32(17)+tx+Int32(1)]
    nothing
end
@kernel function matrix_3!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (272,)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[y*N+x+Int32(1)]=tile[tx*Int32(17)+ty+Int32(1)]
    nothing
end
@kernel function matrix_4!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    @inbounds out[x*N+y+Int32(1)]=input[y*N+x+Int32(1)]
    nothing
end
@kernel function matrix_5!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (256,)
    @inbounds tile[ty*Int32(16)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(16)+ty+Int32(1)]
    nothing
end
@kernel function matrix_6!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (272,)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(17)+ty+Int32(1)]
    nothing
end
@kernel function matrix_7!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (256,)
    @inbounds tile[ty*Int32(16)+(tx⊻ty)+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(16)+(ty⊻tx)+Int32(1)]
    nothing
end
@kernel function matrix_8!(out,input,N)
    lx,ly=@index(Local,NTuple);gx,gy=@index(Group,NTuple);tx=Int32(lx)-Int32(1);ty=Int32(ly)-Int32(1);bx=Int32(gx)-Int32(1);by=Int32(gy)-Int32(1)
    oldbx=bx;bx=(bx+by)%(N÷Int32(16));by=oldbx
    x=bx*Int32(16)+tx;y=by*Int32(16)+ty
    tile=@localmem Float32 (272,)
    @inbounds tile[ty*Int32(17)+tx+Int32(1)]=input[y*N+x+Int32(1)]
    @synchronize
    @inbounds out[(bx*Int32(16)+ty)*N+by*Int32(16)+tx+Int32(1)]=tile[tx*Int32(17)+ty+Int32(1)]
    nothing
end
function study_prepare(out,input,N,kind)
    ctors=(matrix_0!,matrix_1!,matrix_2!,matrix_3!,matrix_4!,matrix_5!,matrix_6!,matrix_7!,matrix_8!);k=ctors[kind+1](CUDABackend(),(16,16))
    ()->k(out,input,Int32(N);ndrange=(N,N))
end
exit(study_main(study_prepare))
