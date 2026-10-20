# Source-level GPU arrays only; CUDA reduction/broadcast internal pools excluded.
const TSNE_G=64
function tsne_guard_array!(storage,::Type{T},n;source=nothing,value=zero(T)) where T
    count=Int(n);base=CuArray{T}(undef,count+2TSNE_G)
    fill!(view(base,1:TSNE_G),T(165));fill!(view(base,count+TSNE_G+1:count+2TSNE_G),T(165))
    payload=unsafe_wrap(CuArray,pointer(base,TSNE_G+1),(count,);own=false)
    if source===nothing;fill!(payload,T(value));else;copyto!(payload,source);end
    push!(storage,(base,count,T));payload
end
function tsne_guard_check(storage,names)
    @assert length(storage)==length(names)
    records=String[]
    for ((base,count,T),name) in zip(storage,names)
        head=collect(reinterpret(UInt8,Array(view(base,1:TSNE_G))))
        tail=collect(reinterpret(UInt8,Array(view(base,count+TSNE_G+1:count+2TSNE_G))))
        expected=collect(reinterpret(UInt8,fill(T(165),TSNE_G)))
        pass=head==expected&&tail==expected
        push!(records,"{\"name\":\"$name\",\"payload_bytes\":$(count*sizeof(T)),\"element_bytes\":$(sizeof(T)),\"guard_elements_each_end\":64,\"expected_bytes\":[$(join(expected,','))],\"head_bytes\":[$(join(head,','))],\"tail_bytes\":[$(join(tail,','))],\"pass\":$pass}")
        pass||error("Source GPU allocation guard changed: ",name)
    end
    "{\"pass\":true,\"scope\":\"All explicit source Julia arrays, excluding CUDA/KA/JACC internaltemporarypools and CPUmemory\",\"allocations\":[$(join(records,','))]}\n"
end
