include(joinpath(@__DIR__,"..","study_common.jl"))
s=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,replace(s,r"main\(\)\s*$"=>""),joinpath(@__DIR__,"main.jl"))
function study_prepare(kc,vc,loc,k,v,batch,row)
    ()->(@cuda threads=256 blocks=cld(batch,8) store_kv_cache_kernel!(kc,vc,loc,UInt64(batch),k,v,UInt64(row),UInt64(row),UInt64(row)))
end
exit(study_main(study_prepare))
