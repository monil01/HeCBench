using CUDA,Libdl,SHA,KernelAbstractions
include(joinpath(@__DIR__,"..","sa_device.jl"))
@kernel function init_entry!(a,n)
    gi=@index(Global,Linear)
    sa_init!(Int32(gi)-Int32(1),a,n)
end
@kernel function key_entry!(a,b,s,n,i)
    gi=@index(Global,Linear)
    sa_key!(Int32(gi)-Int32(1),a,b,s,n,i)
end
@kernel function scan_entry!(s,a,b,n)
    gi=@index(Global,Linear)
    sa_scanflags!(Int32(gi)-Int32(1),s,a,b,n)
end
@kernel function rank_entry!(a,b,c,n,n0)
    gi=@index(Global,Linear)
    sa_setrank!(Int32(gi)-Int32(1),a,b,c,n,n0)
end
@kernel function store_entry!(a,b,n)
    gi=@index(Global,Linear)
    sa_storerank!(Int32(gi)-Int32(1),a,b,n)
end
@kernel function unique_entry!(a,b,n)
    gi=@index(Global,Linear)
    sa_unique!(Int32(gi)-Int32(1),a,b,n)
end
@kernel function mod0_entry!(a,b,n0,n)
    gi=@index(Global,Linear)
    sa_mod0flags!(Int32(gi)-Int32(1),a,b,n0,n)
end
@kernel function set0_entry!(a,b,c,n0,n)
    gi=@index(Global,Linear)
    sa_setmod0!(Int32(gi)-Int32(1),a,b,c,n0,n)
end
@kernel function merge_entry!(a,b,c,s,r,n0,n02,n)
    gi=@index(Global,Linear)
    sa_merge!(Int32(gi)-Int32(1),a,b,c,s,r,n0,n02,n)
end
function sa_launch(::Val{:init},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    init_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:key},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    key_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:scan},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    scan_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:rank},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    rank_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:store},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    store_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:unique},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    unique_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:mod0},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    mod0_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:set0},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    set0_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
function sa_launch(::Val{:merge},n,args...)
    backend=KernelAbstractions.get_backend(args[1])
    merge_entry!(backend,256)(args...;ndrange=cld(Int(n),256)*256)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
sa_study()
