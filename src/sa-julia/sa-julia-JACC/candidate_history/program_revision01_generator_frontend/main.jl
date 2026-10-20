using CUDA,Libdl,SHA,JACC
include(joinpath(@__DIR__,"..","sa_device.jl"))
CUDA.stream!(CUDA.legacy_stream())
function sa_possible_grids()
    grids=Set{Int}()
    for line in eachline(ARGS[1]*"_cases.txt")
        n=parse(Int,split(line)[1])
        while true
            n0=(n+2)÷3;n02=n0+n÷3
            push!(grids,cld(n02,256));push!(grids,cld(n0,256));push!(grids,cld(n0+n02,256))
            n<=1&&break
            n=n02
        end
    end
    sort!(collect(grids))
end
const SA_JACC_SPECS=Dict(b=>JACC.launch_spec(threads=256,blocks=b,shmem_size=0,sync=false)for b in sa_possible_grids())
function init_callback(idx,a,n)
    sa_init!(Int32(idx)-Int32(1),a,n)
    nothing
end
function key_callback(idx,a,b,s,n,i)
    sa_key!(Int32(idx)-Int32(1),a,b,s,n,i)
    nothing
end
function scan_callback(idx,s,a,b,n)
    sa_scanflags!(Int32(idx)-Int32(1),s,a,b,n)
    nothing
end
function rank_callback(idx,a,b,c,n,n0)
    sa_setrank!(Int32(idx)-Int32(1),a,b,c,n,n0)
    nothing
end
function store_callback(idx,a,b,n)
    sa_storerank!(Int32(idx)-Int32(1),a,b,n)
    nothing
end
function unique_callback(idx,a,b,n)
    sa_unique!(Int32(idx)-Int32(1),a,b,n)
    nothing
end
function mod0_callback(idx,a,b,n0,n)
    sa_mod0flags!(Int32(idx)-Int32(1),a,b,n0,n)
    nothing
end
function set0_callback(idx,a,b,c,n0,n)
    sa_setmod0!(Int32(idx)-Int32(1),a,b,c,n0,n)
    nothing
end
function merge_callback(idx,a,b,c,s,r,n0,n02,n)
    sa_merge!(Int32(idx)-Int32(1),a,b,c,s,r,n0,n02,n)
    nothing
end
function sa_launch(::Val{:init},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,init_callback,args...)
    nothing
end
function sa_launch(::Val{:key},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,key_callback,args...)
    nothing
end
function sa_launch(::Val{:scan},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,scan_callback,args...)
    nothing
end
function sa_launch(::Val{:rank},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,rank_callback,args...)
    nothing
end
function sa_launch(::Val{:store},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,store_callback,args...)
    nothing
end
function sa_launch(::Val{:unique},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,unique_callback,args...)
    nothing
end
function sa_launch(::Val{:mod0},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,mod0_callback,args...)
    nothing
end
function sa_launch(::Val{:set0},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,set0_callback,args...)
    nothing
end
function sa_launch(::Val{:merge},n,args...)
    b=cld(Int(n),256)
    JACC.parallel_for(SA_JACC_SPECS[b],b*256,merge_callback,args...)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
sa_study()
