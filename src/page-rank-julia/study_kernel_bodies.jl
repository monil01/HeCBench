const D_FACTOR=0.85f0
@inline function body_map!(i0,pages, page_ranks, maps, noutlinks, n::Int32)
    if i0 < n
        outbound_rank = page_ranks[i0 + Int32(1)] / Float32(noutlinks[i0 + Int32(1)])
        base = i0 * n
        for j0 in Int32(0):(n - Int32(1))
            @inbounds maps[base + j0 + Int32(1)] = Float32(pages[base + j0 + Int32(1)]) * outbound_rank
        end
    end
    return
end

@inline function body_reduce!(j0,page_ranks, maps, n::Int32, diffs)
    if j0 < n
        old_rank = page_ranks[j0 + Int32(1)]
        new_rank = 0.0f0
        for i0 in Int32(0):(n - Int32(1))
            @inbounds new_rank += maps[i0 * n + j0 + Int32(1)]
        end
        new_rank = ((1.0f0 - D_FACTOR) / Float32(n)) + D_FACTOR * new_rank
        @inbounds diffs[j0 + Int32(1)] = max(abs(new_rank - old_rank), diffs[j0 + Int32(1)])
        @inbounds page_ranks[j0 + Int32(1)] = new_rank
    end
    return
end

