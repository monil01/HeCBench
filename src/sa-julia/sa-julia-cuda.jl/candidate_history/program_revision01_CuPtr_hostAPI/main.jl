using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"..","sa_device.jl"))
@inline sa_index()=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
function init_entry!(a,n);sa_init!(sa_index(),a,n);nothing;end
function key_entry!(a,b,s,n,i);sa_key!(sa_index(),a,b,s,n,i);nothing;end
function scan_entry!(s,a,b,n);sa_scanflags!(sa_index(),s,a,b,n);nothing;end
function rank_entry!(a,b,c,n,n0);sa_setrank!(sa_index(),a,b,c,n,n0);nothing;end
function store_entry!(a,b,n);sa_storerank!(sa_index(),a,b,n);nothing;end
function unique_entry!(a,b,n);sa_unique!(sa_index(),a,b,n);nothing;end
function mod0_entry!(a,b,n0,n);sa_mod0flags!(sa_index(),a,b,n0,n);nothing;end
function set0_entry!(a,b,c,n0,n);sa_setmod0!(sa_index(),a,b,c,n0,n);nothing;end
function merge_entry!(a,b,c,s,r,n0,n02,n);sa_merge!(sa_index(),a,b,c,s,r,n0,n02,n);nothing;end
function sa_launch(::Val{:init},n,args...);@cuda threads=256 blocks=cld(n,256) init_entry!(args...);nothing;end
function sa_launch(::Val{:key},n,args...);@cuda threads=256 blocks=cld(n,256) key_entry!(args...);nothing;end
function sa_launch(::Val{:scan},n,args...);@cuda threads=256 blocks=cld(n,256) scan_entry!(args...);nothing;end
function sa_launch(::Val{:rank},n,args...);@cuda threads=256 blocks=cld(n,256) rank_entry!(args...);nothing;end
function sa_launch(::Val{:store},n,args...);@cuda threads=256 blocks=cld(n,256) store_entry!(args...);nothing;end
function sa_launch(::Val{:unique},n,args...);@cuda threads=256 blocks=cld(n,256) unique_entry!(args...);nothing;end
function sa_launch(::Val{:mod0},n,args...);@cuda threads=256 blocks=cld(n,256) mod0_entry!(args...);nothing;end
function sa_launch(::Val{:set0},n,args...);@cuda threads=256 blocks=cld(n,256) set0_entry!(args...);nothing;end
function sa_launch(::Val{:merge},n,args...);@cuda threads=256 blocks=cld(n,256) merge_entry!(args...);nothing;end
include(joinpath(@__DIR__,"..","study_common.jl"))
sa_study()
