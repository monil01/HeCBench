using CUDA,Libdl,SHA,JACC
# Source CSR/CSC/level indices remain zero-based UInt32/Int32 in device storage.
# Native source float arithmetic and full-block geometry retained; CUDA-specific
# warp synchronization, shared storage and Float32 atomics are explicit backend use.
@inline function slu_rl_body!(a,r,v,d,rp,ci,rd,lev,tmp,n,head,pos,pert,::Val{P}) where P
    tid=Int32(threadIdx().x)-Int32(1);bid=Int32(blockIdx().x)-Int32(1)
    nt=Int32(blockDim().x);wid=tid÷Int32(32);lane=tid%Int32(32)
    @inbounds begin
        col=lev[head+pos+bid+Int32(1)];diag=Int32(d[col+Int32(1)])
        size=Int32(a[col+Int32(2)])-diag-Int32(1)
        shared=CUDA.CuDynamicSharedArray(Float32,nt÷Int32(32))
        if P
            if tid==0 && size>0 && abs(v[diag+Int32(1)])<pert
                v[diag+Int32(1)]=pert
            end
            CUDA.sync_threads()
        end
        off=Int32(0)
        while size>off
            if tid+off<size
                e=diag+tid+off+Int32(1);row=Int32(r[e+Int32(1)])
                value=v[e+Int32(1)]/v[diag+Int32(1)]
                v[e+Int32(1)]=value;tmp[bid*n+row+Int32(1)]=value
            end
            off+=nt
        end
        CUDA.sync_threads()
        subpos=Int32(rd[col+Int32(1)])+wid+Int32(1)
        subsize=Int32(rp[col+Int32(2)])-Int32(rd[col+Int32(1)])-Int32(1)
        wo=Int32(0)
        while subsize>wo
            if wid+wo<subsize
                subcol=Int32(ci[subpos+wo+Int32(1)])
                start=Int32(a[subcol+Int32(1)]);len=Int32(a[subcol+Int32(2)])-start
                off=Int32(0)
                while off<len
                    active=lane+off<len;e=start+lane+off;row=Int32(0)
                    if active
                        row=Int32(r[e+Int32(1)])
                        if row==col;shared[wid+Int32(1)]=v[e+Int32(1)];end
                    end
                    CUDA.sync_warp()
                    if active && row>col
                        CUDA.atomic_add!(pointer(v,e+Int32(1)),-tmp[row+n*bid+Int32(1)]*shared[wid+Int32(1)])
                    end
                    CUDA.sync_warp();off+=Int32(32)
                end
            end
            wo+=nt÷Int32(32)
        end
        CUDA.sync_threads();off=Int32(0)
        while size>off
            if tid+off<size
                e=diag+tid+off+Int32(1);row=Int32(r[e+Int32(1)])
                tmp[bid*n+row+Int32(1)]=0f0
            end
            off+=nt
        end
    end
    nothing
end
function slu_rl!(i,a,r,v,d,rp,ci,rd,lev,tmp,n,col,head,pos,slot,pert)
    slu_rl_body!(a,r,v,d,rp,ci,rd,lev,tmp,n,head,pos,pert,Val(false));nothing
end
function slu_rl_perturb!(i,a,r,v,d,rp,ci,rd,lev,tmp,n,col,head,pos,slot,pert)
    slu_rl_body!(a,r,v,d,rp,ci,rd,lev,tmp,n,head,pos,pert,Val(true));nothing
end
@inline function slu_factor_body!(a,r,v,d,tmp,n,col,slot,pert,::Val{P}) where P
    tid=Int32(threadIdx().x)-Int32(1);nt=Int32(blockDim().x)
    @inbounds begin
        diag=Int32(d[col+Int32(1)]);len=Int32(a[col+Int32(2)])-diag-Int32(1)
        if P
            if tid==0 && len>0 && abs(v[diag+Int32(1)])<pert;v[diag+Int32(1)]=pert;end
            CUDA.sync_threads()
        end
        off=Int32(0)
        while len>off
            if tid+off<len
                e=diag+tid+off+Int32(1);row=Int32(r[e+Int32(1)])
                x=v[e+Int32(1)]/v[diag+Int32(1)];v[e+Int32(1)]=x;tmp[slot*n+row+Int32(1)]=x
            end
            off+=nt
        end
    end
    nothing
end
function slu_factor!(i,a,r,v,d,rp,ci,rd,lev,tmp,n,col,head,pos,slot,pert)
    slu_factor_body!(a,r,v,d,tmp,n,col,slot,pert,Val(false));nothing
end
function slu_factor_perturb!(i,a,r,v,d,rp,ci,rd,lev,tmp,n,col,head,pos,slot,pert)
    slu_factor_body!(a,r,v,d,tmp,n,col,slot,pert,Val(true));nothing
end
function slu_update!(i,a,r,v,d,rp,ci,rd,lev,tmp,n,col,head,pos,slot,pert)
    tid=Int32(threadIdx().x)-Int32(1);bid=Int32(blockIdx().x)-Int32(1);nt=Int32(blockDim().x)
    shared=CUDA.CuStaticSharedArray(Float32,1)
    @inbounds begin
        subpos=Int32(rd[col+Int32(1)])+bid+Int32(1);subcol=Int32(ci[subpos+Int32(1)])
        start=Int32(a[subcol+Int32(1)]);len=Int32(a[subcol+Int32(2)])-start;off=Int32(0)
        while off<len
            active=tid+off<len;e=start+tid+off;row=Int32(0)
            if active
                row=Int32(r[e+Int32(1)]);if row==col;shared[1]=v[e+Int32(1)];end
            end
            CUDA.sync_threads()
            if active && row>col
                CUDA.atomic_add!(pointer(v,e+Int32(1)),-tmp[slot*n+row+Int32(1)]*shared[1])
            end
            CUDA.sync_threads();off+=nt
        end
    end
    nothing
end
function slu_clear!(i,a,r,v,d,rp,ci,rd,lev,tmp,n,col,head,pos,slot,pert)
    tid=Int32(threadIdx().x)-Int32(1);nt=Int32(blockDim().x)
    @inbounds begin
        diag=Int32(d[col+Int32(1)]);len=Int32(a[col+Int32(2)])-diag-Int32(1);off=Int32(0)
        while len>off
            if tid+off<len
                e=diag+tid+off+Int32(1);row=Int32(r[e+Int32(1)]);tmp[slot*n+row+Int32(1)]=0f0
            end
            off+=nt
        end
    end
    nothing
end
# Every launch is an exact multiple of the source block width: all source
# collective participants enter JACC's per-index function, including sparse tails.
# Complete geometry keys prevent mutation/reuse of a spec for a different grid.
const SLU_JACC_SPECS=Dict{NTuple{4,Int},Any}()
function slu_spec(q)
    key=(Int(q.op),Int(q.threads),Int(q.blocks),Int(q.smem))
    get!(SLU_JACC_SPECS,key) do
        JACC.launch_spec(;threads=key[2],blocks=key[3],shmem_size=key[4],sync=false,stream=CUDA.legacy_stream())
    end
end
function slu_launch(q,a,r,v,d,rp,ci,rd,lev,tmp)
    kernels=(slu_rl!,slu_rl_perturb!,slu_factor!,slu_factor_perturb!,slu_update!,slu_clear!)
    op=Int(q.op)+1;1<=op<=6||error("Unknownsourceentry")
    spec=slu_spec(q);range=Int(q.threads)*Int(q.blocks)
    JACC.parallel_for(spec,range,kernels[op],a,r,v,d,rp,ci,rd,lev,tmp,Int32(q.n),Int32(q.col),q.head,q.pos,q.slot,q.pert)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
slu_study()
