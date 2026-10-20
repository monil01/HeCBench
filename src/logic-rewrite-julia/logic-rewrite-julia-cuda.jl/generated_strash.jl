# GENERATED from original source; full-pipeline verification PENDING.
# Original buildHashTable source device body; candidate, not certified.
@inline function lr_buildHashTable(ctx,pFanin0,pFanin1,htKeys,htValues,htCapacity,nNodes,nPIs)::Nothing
    idx_L57=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local lit0_L58
    local lit1_L58
    local temp_L58
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L57) < lr_cast(Int32,nNodes))))
        id_L60=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,idx_L57) + lr_cast(Int32,nPIs))) + lr_cast(Int32,1)))))
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(UInt32,id_L60)])); lit0_L58 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(UInt32,id_L60)])); lit1_L58 = _lr_rhs; _lr_rhs))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit0_L58) > lr_cast(Int32,lit1_L58))))
            (((_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit0_L58)); temp_L58 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit1_L58)); lit0_L58 = _lr_rhs; _lr_rhs)); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L58)); lit1_L58 = _lr_rhs; _lr_rhs))
        end
        key_L64=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,lit0_L58),lr_cast(Int32,lit1_L58)))
        lr_s_insert_single_no_update(lr_pointer_cast(UInt64,htKeys),lr_pointer_cast(UInt32,htValues),lr_cast(UInt64,key_L64),lr_cast(UInt32,id_L60),lr_cast(Int32,htCapacity))
    end
    
    return nothing
end

# Original markReadyNodes source device body; candidate, not certified.
@inline function lr_markReadyNodes(ctx,pFanin0,pFanin1,vRemainNodes,vOld2NewLit,vMarks,nRemain)::Nothing
    idx_L72=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local nodeId_L73
    local id0_L74
    local id1_L74
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L72) < lr_cast(Int32,nRemain))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vRemainNodes))[lr_cast(Int32,idx_L72)])); nodeId_L73 = _lr_rhs; _lr_rhs)
        ((_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId_L73)]))); id0_L74 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId_L73)]))); id1_L74 = _lr_rhs; _lr_rhs))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,id0_L74)]) != lr_cast(Int32,(-lr_cast(Int32,1)))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,id1_L74)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vMarks))[lr_cast(Int32,idx_L72)] = _lr_rhs; _lr_rhs)
        end
    end
    
    return nothing
end

# Original insertLevelNodes source device body; candidate, not certified.
@inline function lr_insertLevelNodes(ctx,vReadyNodes,vOld2NewLit,pFanin0,pFanin1,htKeys,htValues,htCapacity,vKeysBuffer,idCounter,nReady)::Nothing
    idx_L89=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local nodeId_L90
    local lit0_L91
    local lit1_L91
    local id0_L91
    local id1_L91
    local temp_L91
    local temp0_L92
    local key_L93
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L89) < lr_cast(Int32,nReady))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vReadyNodes))[lr_cast(Int32,idx_L89)])); nodeId_L90 = _lr_rhs; _lr_rhs)
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId_L90)])); lit0_L91 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId_L90)])); lit1_L91 = _lr_rhs; _lr_rhs))
        ((_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lit0_L91))); id0_L91 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lit1_L91))); id1_L91 = _lr_rhs; _lr_rhs))
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeNotCond(lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,id0_L91)]),lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L91)))); lit0_L91 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeNotCond(lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,id1_L91)]),lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L91)))); lit1_L91 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit0_L91) > lr_cast(Int32,lit1_L91))))
            (((_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit0_L91)); temp_L91 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit1_L91)); lit0_L91 = _lr_rhs; _lr_rhs)); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L91)); lit1_L91 = _lr_rhs; _lr_rhs))
        end
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_s_AigNodeID(lr_cast(Int32,lit0_L91)) < lr_cast(Int32,(lr_cast(Int32,idCounter) + lr_cast(Int32,idx_L89))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_s_AigNodeID(lr_cast(Int32,lit1_L91)) < lr_cast(Int32,(lr_cast(Int32,idCounter) + lr_cast(Int32,idx_L89))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,lit0_L91),lr_cast(Int32,lit1_L91))); key_L93 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,key_L93)); (lr_pointer_cast(UInt64,vKeysBuffer))[lr_cast(Int32,idx_L89)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_s_checkTrivialAndCases(lr_cast(Int32,lit0_L91),lr_cast(Int32,lit1_L91))); temp0_L92 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,temp0_L92) == lr_cast(UInt32,typemax(UInt32)))))
            lr_s_insert_single_no_update(lr_pointer_cast(UInt64,htKeys),lr_pointer_cast(UInt32,htValues),lr_cast(UInt64,key_L93),lr_cast(UInt32,lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,idCounter) + lr_cast(Int32,idx_L89)))))),lr_cast(Int32,htCapacity))
        end
    end
    
    return nothing
end

# Original updateLevelNodesNewIds source device body; candidate, not certified.
@inline function lr_updateLevelNodesNewIds(ctx,vReadyNodes,vOld2NewLit,vKeysBuffer,htKeys,htValues,htCapacity,nReady)::Nothing
    _lr_mem_0=lr_alloc(UInt32)
    lit0_L127=_lr_mem_0
    _lr_mem_1=lr_alloc(UInt32)
    lit1_L127=_lr_mem_1
    idx_L125=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local nodeId_L126


    local temp0_L128
    local key_L129
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L125) < lr_cast(Int32,nReady))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vReadyNodes))[lr_cast(Int32,idx_L125)])); nodeId_L126 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(lr_pointer_cast(UInt64,vKeysBuffer))[lr_cast(Int32,idx_L125)])); key_L129 = _lr_rhs; _lr_rhs)
        lr_s_unbindAndNodeKeys(lr_cast(UInt64,key_L129),lit0_L127,lit1_L127)
        (_lr_rhs=lr_cast(UInt32,lr_s_checkTrivialAndCases(lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,lit0_L127[0]))),lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,lit1_L127[0]))))); temp0_L128 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,temp0_L128) == lr_cast(UInt32,typemax(UInt32)))))
            (_lr_rhs=lr_cast(UInt32,lr_s_retrieve_single(lr_pointer_cast(UInt64,htKeys),lr_pointer_cast(UInt32,htValues),lr_cast(UInt64,key_L129),lr_cast(Int32,htCapacity))); temp0_L128 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,temp0_L128) << lr_cast(Int32,1)))); temp0_L128 = _lr_rhs; _lr_rhs)
        end
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,nodeId_L126)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,temp0_L128))); (lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,nodeId_L126)] = _lr_rhs; _lr_rhs)
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original assignOld2NewConsecutiveIds source device body; candidate, not certified.
@inline function lr_assignOld2NewConsecutiveIds(ctx,vOldIds,vOld2NewIdConsecutive,nPIs,nObjsNew)::Nothing
    idx_L150=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L150) <= lr_cast(Int32,(lr_cast(Int32,nPIs) + lr_cast(Int32,1))))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idx_L150)); (lr_pointer_cast(Int32,vOld2NewIdConsecutive))[lr_cast(Int32,idx_L150)] = _lr_rhs; _lr_rhs)
    else
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L150) < lr_cast(Int32,nObjsNew))))
            nodeIdOld_L154=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,vOldIds))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,idx_L150) - lr_cast(Int32,nPIs))) - lr_cast(Int32,1)))]))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,nodeIdOld_L154) > lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,nPIs))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idx_L150)); (lr_pointer_cast(Int32,vOld2NewIdConsecutive))[lr_cast(UInt32,nodeIdOld_L154)] = _lr_rhs; _lr_rhs)
        end
    end
    
    return nothing
end

# Original unbindKeysReId source device body; candidate, not certified.
@inline function lr_unbindKeysReId(ctx,vOldKeys,vOld2NewIdConsecutive,vFanin0New,vFanin1New,vNumFanoutsNew,nNodesNew)::Nothing
    _lr_mem_0=lr_alloc(UInt32)
    oldLit0_L167=_lr_mem_0
    _lr_mem_1=lr_alloc(UInt32)
    oldLit1_L167=_lr_mem_1
    idx_L165=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L165) < lr_cast(Int32,nNodesNew))))


        local oldId0_L168
        local oldId1_L168
        local newId0_L168
        local newId1_L168
        lr_s_unbindAndNodeKeys(lr_cast(UInt64,(lr_pointer_cast(UInt64,vOldKeys))[lr_cast(Int32,idx_L165)]),oldLit0_L167,oldLit1_L167)
        ((_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,oldLit0_L167[0]))))); oldId0_L168 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,oldLit1_L167[0]))))); oldId1_L168 = _lr_rhs; _lr_rhs))
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewIdConsecutive))[lr_cast(Int32,oldId0_L168)])); newId0_L168 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewIdConsecutive))[lr_cast(Int32,oldId1_L168)])); newId1_L168 = _lr_rhs; _lr_rhs))
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,newId0_L168),lr_s_AigNodeIsComplement(lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,oldLit0_L167[0])))))); (lr_pointer_cast(Int32,vFanin0New))[lr_cast(Int32,idx_L165)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,newId1_L168),lr_s_AigNodeIsComplement(lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,oldLit1_L167[0])))))); (lr_pointer_cast(Int32,vFanin1New))[lr_cast(Int32,idx_L165)] = _lr_rhs; _lr_rhs)
        lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(Int32,newId0_L168)),lr_cast(Int32,1))
        lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(Int32,newId1_L168)),lr_cast(Int32,1))
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original poReId source device body; candidate, not certified.
@inline function lr_poReId(ctx,vOld2NewIdConsecutive,vOutsNew,vNumFanoutsNew,nPOs)::Nothing
    idx_L184=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L184) < lr_cast(Int32,nPOs))))
        local oldLit_L186
        local oldId_L186
        local newId_L186
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vOutsNew))[lr_cast(Int32,idx_L184)])); oldLit_L186 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,oldLit_L186))); oldId_L186 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewIdConsecutive))[lr_cast(Int32,oldId_L186)])); newId_L186 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,newId_L186),lr_s_AigNodeIsComplement(lr_cast(Int32,oldLit_L186)))); (lr_pointer_cast(Int32,vOutsNew))[lr_cast(Int32,idx_L184)] = _lr_rhs; _lr_rhs)
        lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(Int32,newId_L186)),lr_cast(Int32,1))
    end
    
    return nothing
end

# Original poUpdateId source device body; candidate, not certified.
@inline function lr_poUpdateId(ctx,pOuts,vOld2NewLit,vOutsNew,nPOs)::Nothing
    idx_L200=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L200) < lr_cast(Int32,nPOs))))
        local oldLit_L202
        local oldId_L202
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pOuts))[lr_cast(Int32,idx_L200)])); oldLit_L202 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,oldLit_L202))); oldId_L202 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeNotCond(lr_cast(Int32,(lr_pointer_cast(Int32,vOld2NewLit))[lr_cast(Int32,oldId_L202)]),lr_s_AigNodeIsComplement(lr_cast(Int32,oldLit_L202)))); (lr_pointer_cast(Int32,vOutsNew))[lr_cast(Int32,idx_L200)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original markDanglingNodesIter source device body; candidate, not certified.
@inline function lr_markDanglingNodesIter(ctx,pFanin0,pFanin1,pNumFanouts,vDanglingMarks,nNodes,nPIs)::Nothing
    idx_L212=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L212) < lr_cast(Int32,nNodes))))
        nodeId_L214=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,idx_L212) + lr_cast(Int32,nPIs))) + lr_cast(Int32,1))))
        local id0_L214
        local id1_L214
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,nodeId_L214)]) >= lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        if lr_truth((lr_truth((!lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_pointer_cast(Int32,vDanglingMarks))[lr_cast(Int32,idx_L212)]))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,nodeId_L214)]) == lr_cast(Int32,0))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vDanglingMarks))[lr_cast(Int32,idx_L212)] = _lr_rhs; _lr_rhs)
            ((_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId_L214)]))); id0_L214 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId_L214)]))); id1_L214 = _lr_rhs; _lr_rhs))
            lr_atomicSub((lr_pointer_cast(Int32,pNumFanouts) + lr_cast(Int32,id0_L214)),lr_cast(Int32,1))
            lr_atomicSub((lr_pointer_cast(Int32,pNumFanouts) + lr_cast(Int32,id1_L214)),lr_cast(Int32,1))
        end
    end
    
    return nothing
end
