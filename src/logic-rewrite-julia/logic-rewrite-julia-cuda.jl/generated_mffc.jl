# GENERATED from original source; full-pipeline verification PENDING.
# Original findReconvMFFCCut_iter source device body; candidate, not certified.
@inline function lr_findReconvMFFCCut_iter(ctx,rootId,pFanin0,pFanin1,pNumFanouts,pLevels,vCutTable,vCutSizes,travIds,travRefs,pTravSize,pConeSize,nPIs,nMaxCutSize)::Int32
    local nodeId_L14
    local faninId_L14
    bestId_L14=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    bestIdx_L14=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    local faninLoc_L14
    bestCost_L15=lr_cast(Int32,lr_cast(Int32,100))
    local currCost_L15
    local fFanin0Visited_L16
    local fFanin1Visited_L16
    local nodeRef_L17
    vCutRoot_L18=lr_pointer_cast(Int32,(lr_pointer_cast(Int32,vCutTable) + lr_cast(Int32,(lr_cast(Int32,rootId) * lr_cast(Int32,16)))))
    i_L22=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L22) < lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutRoot_L18))[lr_cast(Int32,i_L22)])); nodeId_L14 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-Int32(1)))); nodeRef_L17 = _lr_rhs; _lr_rhs)
        j_L30=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L30) < lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,travIds))[lr_cast(Int32,j_L30)]) == lr_cast(Int32,nodeId_L14))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,j_L30)])); nodeRef_L17 = _lr_rhs; _lr_rhs)
                break
            end
            (_lr_old=j_L30; j_L30=lr_cast(Int32,(j_L30+Int32(1))); _lr_old)
        end
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,nodeRef_L17) != lr_cast(Int32,((-Int32(1)))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        if lr_truth((lr_truth(lr_cast(Bool,lr_s_AigIsPIConst(lr_cast(Int32,nodeId_L14),lr_cast(Int32,nPIs)))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,nodeRef_L17) != lr_cast(Int32,0))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,999)); currCost_L15 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); fFanin1Visited_L16 = _lr_rhs; _lr_rhs))); fFanin0Visited_L16 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId_L14)]))); faninId_L14 = _lr_rhs; _lr_rhs)
            j_L50=lr_cast(Int32,lr_cast(Int32,0))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L50) < lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]))))
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,travIds))[lr_cast(Int32,j_L50)]) == lr_cast(Int32,faninId_L14))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); fFanin0Visited_L16 = _lr_rhs; _lr_rhs)
                    break
                end
                (_lr_old=j_L50; j_L50=lr_cast(Int32,(j_L50+Int32(1))); _lr_old)
            end
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId_L14)]))); faninId_L14 = _lr_rhs; _lr_rhs)
            j_L62=lr_cast(Int32,lr_cast(Int32,0))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L62) < lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]))))
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,travIds))[lr_cast(Int32,j_L62)]) == lr_cast(Int32,faninId_L14))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); fFanin1Visited_L16 = _lr_rhs; _lr_rhs)
                    break
                end
                (_lr_old=j_L62; j_L62=lr_cast(Int32,(j_L62+Int32(1))); _lr_old)
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,1) - lr_cast(Int32,fFanin0Visited_L16)))) + (lr_cast(Int32,(lr_cast(Int32,1) - lr_cast(Int32,fFanin1Visited_L16))))))); currCost_L15 = _lr_rhs; _lr_rhs)
        end
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,bestCost_L15) > lr_cast(Int32,currCost_L15)))) || lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,bestCost_L15) == lr_cast(Int32,currCost_L15)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,pLevels))[lr_cast(Int32,nodeId_L14)]) > lr_cast(Int32,(lr_pointer_cast(Int32,pLevels))[lr_cast(Int32,bestId_L14)])))))))))
            (((_lr_rhs=lr_cast(Int32,lr_cast(Int32,currCost_L15)); bestCost_L15 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeId_L14)); bestId_L14 = _lr_rhs; _lr_rhs)); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L22)); bestIdx_L14 = _lr_rhs; _lr_rhs))
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,bestCost_L15) == lr_cast(Int32,0))))
            break
        end
        (_lr_old=i_L22; i_L22=lr_cast(Int32,(i_L22+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,bestId_L14) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,bestCost_L15) < lr_cast(Int32,3))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]) - lr_cast(Int32,1))) + lr_cast(Int32,bestCost_L15))) > lr_cast(Int32,nMaxCutSize))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,bestId_L14),lr_cast(Int32,nPIs))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    i_L89=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,bestIdx_L14) + lr_cast(Int32,1))))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L89) < lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutRoot_L18))[lr_cast(Int32,i_L89)])); (lr_pointer_cast(Int32,vCutRoot_L18))[lr_cast(Int32,(lr_cast(Int32,i_L89) - lr_cast(Int32,1)))] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L89; i_L89=lr_cast(Int32,(i_L89+Int32(1))); _lr_old)
    end
    (_lr_old=(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]; (lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]=lr_cast(Int32,((lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]-Int32(1))); _lr_old)
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,bestId_L14)]))); faninId_L14 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); faninLoc_L14 = _lr_rhs; _lr_rhs)
    j_L100=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L100) < lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,travIds))[lr_cast(Int32,j_L100)]) == lr_cast(Int32,faninId_L14))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,j_L100)); faninLoc_L14 = _lr_rhs; _lr_rhs)
            break
        end
        (_lr_old=j_L100; j_L100=lr_cast(Int32,(j_L100+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,faninLoc_L14) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L14)); (lr_pointer_cast(Int32,vCutRoot_L18))[(_lr_old=(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]; (lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]=lr_cast(Int32,((lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]) < lr_cast(Int32,192))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0])); faninLoc_L14 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L14)); (lr_pointer_cast(Int32,travIds))[lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0])] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,faninId_L14)])); (lr_pointer_cast(Int32,travRefs))[(_lr_old=((lr_pointer_cast(Int32,pTravSize))[0]); ((lr_pointer_cast(Int32,pTravSize))[0])=lr_cast(Int32,(((lr_pointer_cast(Int32,pTravSize))[0])+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,faninLoc_L14) != lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    end
    (_lr_old=(lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,faninLoc_L14)]; (lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,faninLoc_L14)]=lr_cast(Int32,((lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,faninLoc_L14)]-Int32(1))); _lr_old)
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,bestId_L14)]))); faninId_L14 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); faninLoc_L14 = _lr_rhs; _lr_rhs)
    j_L129=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L129) < lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,travIds))[lr_cast(Int32,j_L129)]) == lr_cast(Int32,faninId_L14))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,j_L129)); faninLoc_L14 = _lr_rhs; _lr_rhs)
            break
        end
        (_lr_old=j_L129; j_L129=lr_cast(Int32,(j_L129+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,faninLoc_L14) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L14)); (lr_pointer_cast(Int32,vCutRoot_L18))[(_lr_old=(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]; (lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]=lr_cast(Int32,((lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0]) < lr_cast(Int32,192))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0])); faninLoc_L14 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L14)); (lr_pointer_cast(Int32,travIds))[lr_cast(Int32,(lr_pointer_cast(Int32,pTravSize))[0])] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,faninId_L14)])); (lr_pointer_cast(Int32,travRefs))[(_lr_old=((lr_pointer_cast(Int32,pTravSize))[0]); ((lr_pointer_cast(Int32,pTravSize))[0])=lr_cast(Int32,(((lr_pointer_cast(Int32,pTravSize))[0])+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,faninLoc_L14) != lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    end
    (_lr_old=(lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,faninLoc_L14)]; (lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,faninLoc_L14)]=lr_cast(Int32,((lr_pointer_cast(Int32,travRefs))[lr_cast(Int32,faninLoc_L14)]-Int32(1))); _lr_old)
    (_lr_old=((lr_pointer_cast(Int32,pConeSize))[0]); ((lr_pointer_cast(Int32,pConeSize))[0])=lr_cast(Int32,(((lr_pointer_cast(Int32,pConeSize))[0])+Int32(1))); _lr_old)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]) <= lr_cast(Int32,nMaxCutSize))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    
    return nothing
end

# Original findReconvMFFCCut source device body; candidate, not certified.
@inline function lr_findReconvMFFCCut(ctx,rootId,pFanin0,pFanin1,pNumFanouts,pLevels,vCutTable,vCutSizes,nPIs,nMaxCutSize)::Int32
    _lr_mem_0=lr_alloc(LRArray{Int32,192})
    travIds_L172=lr_pointer_cast(Int32,_lr_mem_0)
    _lr_mem_1=lr_alloc(LRArray{Int32,192})
    travRefs_L172=lr_pointer_cast(Int32,_lr_mem_1)
    _lr_mem_2=lr_alloc(Int32)
    travSize_L173=_lr_mem_2
    _lr_mem_3=lr_alloc(Int32)
    coneSize_L174=_lr_mem_3
    local faninId_L171


    travSize_L173[0]=lr_cast(Int32,lr_cast(Int32,0))
    coneSize_L174[0]=lr_cast(Int32,lr_cast(Int32,0))
    vCutRoot_L175=lr_pointer_cast(Int32,(lr_pointer_cast(Int32,vCutTable) + lr_cast(Int32,(lr_cast(Int32,rootId) * lr_cast(Int32,16)))))
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)] = _lr_rhs; _lr_rhs)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,rootId),lr_cast(Int32,nPIs))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rootId)); (lr_pointer_cast(Int32,travIds_L172))[lr_cast(Int32,travSize_L173[0])] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,travRefs_L172))[(_lr_old=travSize_L173[0]; travSize_L173[0]=lr_cast(Int32,(travSize_L173[0]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,rootId)]))); faninId_L171 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L171)); (lr_pointer_cast(Int32,vCutRoot_L175))[(_lr_old=(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]; (lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]=lr_cast(Int32,((lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L171)); (lr_pointer_cast(Int32,travIds_L172))[lr_cast(Int32,travSize_L173[0])] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,faninId_L171)]) - lr_cast(Int32,1)))); (lr_pointer_cast(Int32,travRefs_L172))[(_lr_old=travSize_L173[0]; travSize_L173[0]=lr_cast(Int32,(travSize_L173[0]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,rootId)]))); faninId_L171 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L171)); (lr_pointer_cast(Int32,vCutRoot_L175))[(_lr_old=(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]; (lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]=lr_cast(Int32,((lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,faninId_L171)); (lr_pointer_cast(Int32,travIds_L172))[lr_cast(Int32,travSize_L173[0])] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,faninId_L171)]) - lr_cast(Int32,1)))); (lr_pointer_cast(Int32,travRefs_L172))[(_lr_old=travSize_L173[0]; travSize_L173[0]=lr_cast(Int32,(travSize_L173[0]+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); coneSize_L174[0] = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,lr_findReconvMFFCCut_iter(ctx ,lr_cast(Int32,rootId),lr_pointer_cast(Int32,pFanin0),lr_pointer_cast(Int32,pFanin1),lr_pointer_cast(Int32,pNumFanouts),lr_pointer_cast(Int32,pLevels),lr_pointer_cast(Int32,vCutTable),lr_pointer_cast(Int32,vCutSizes),lr_pointer_cast(Int32,travIds_L172),lr_pointer_cast(Int32,travRefs_L172),travSize_L173,coneSize_L174,lr_cast(Int32,nPIs),lr_cast(Int32,nMaxCutSize))))

    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId)]) <= lr_cast(Int32,nMaxCutSize))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,coneSize_L174[0])); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base)))
    return nothing
end

# Original cutTruthIter source device body; candidate, not certified.
@inline function lr_cutTruthIter(ctx,nodeId,pFanin0,pFanin1,vTruthMem,visited,pVisitedSize,nWords)::Nothing
    local fanin0Id_L12
    local fanin1Id_L12
    local fanin0TruthIdx_L12
    local fanin1TruthIdx_L12
    local lit0_L13
    local lit1_L13
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId)]))); fanin0Id_L12 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId)]))); fanin1Id_L12 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); fanin1TruthIdx_L12 = _lr_rhs; _lr_rhs))); fanin0TruthIdx_L12 = _lr_rhs; _lr_rhs)
    j_L18=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L18) < lr_cast(Int32,(lr_pointer_cast(Int32,pVisitedSize))[0]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,visited))[lr_cast(Int32,j_L18)]) == lr_cast(Int32,fanin0Id_L12))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,j_L18)); fanin0TruthIdx_L12 = _lr_rhs; _lr_rhs)
            break
        end
        (_lr_old=j_L18; j_L18=lr_cast(Int32,(j_L18+Int32(1))); _lr_old)
    end
    j_L23=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L23) < lr_cast(Int32,(lr_pointer_cast(Int32,pVisitedSize))[0]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,visited))[lr_cast(Int32,j_L23)]) == lr_cast(Int32,fanin1Id_L12))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,j_L23)); fanin1TruthIdx_L12 = _lr_rhs; _lr_rhs)
            break
        end
        (_lr_old=j_L23; j_L23=lr_cast(Int32,(j_L23+Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,(lr_truth(lr_cast(Bool,(lr_cast(Int32,fanin0TruthIdx_L12) != lr_cast(Int32,(-lr_cast(Int32,1)))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,fanin1TruthIdx_L12) != lr_cast(Int32,(-lr_cast(Int32,1))))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId)])); lit0_L13 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId)])); lit1_L13 = _lr_rhs; _lr_rhs))
    if lr_truth((lr_truth((!lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L13)))))) && lr_truth((!lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L13))))))))
        i_L32=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L32) < lr_cast(Int32,nWords))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin0TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L32)))]) & lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin1TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L32)))])))); (lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(Int32,pVisitedSize))[0])) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L32)))] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L32; i_L32=lr_cast(Int32,(i_L32+Int32(1))); _lr_old)
        end
    else
        if lr_truth((lr_truth((!lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L13)))))) && lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L13))))))
            i_L36=lr_cast(Int32,lr_cast(Int32,0))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L36) < lr_cast(Int32,nWords))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin0TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L36)))]) & lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin1TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L36)))])))))); (lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(Int32,pVisitedSize))[0])) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L36)))] = _lr_rhs; _lr_rhs)
                (_lr_old=i_L36; i_L36=lr_cast(Int32,(i_L36+Int32(1))); _lr_old)
            end
        else
            if lr_truth((lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L13)))) && lr_truth((!lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L13))))))))
                i_L40=lr_cast(Int32,lr_cast(Int32,0))
                while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L40) < lr_cast(Int32,nWords))))
                    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin0TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L40)))]))) & lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin1TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L40)))])))); (lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(Int32,pVisitedSize))[0])) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L40)))] = _lr_rhs; _lr_rhs)
                    (_lr_old=i_L40; i_L40=lr_cast(Int32,(i_L40+Int32(1))); _lr_old)
                end
            else
                i_L44=lr_cast(Int32,lr_cast(Int32,0))
                while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L44) < lr_cast(Int32,nWords))))
                    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin0TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L44)))]))) & lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,fanin1TruthIdx_L12) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L44)))])))))); (lr_pointer_cast(UInt32,vTruthMem))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(Int32,pVisitedSize))[0])) * lr_cast(Int32,nWords))) + lr_cast(Int32,i_L44)))] = _lr_rhs; _lr_rhs)
                    (_lr_old=i_L44; i_L44=lr_cast(Int32,(i_L44+Int32(1))); _lr_old)
                end
            end
        end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeId)); (lr_pointer_cast(Int32,visited))[(_lr_old=((lr_pointer_cast(Int32,pVisitedSize))[0]); ((lr_pointer_cast(Int32,pVisitedSize))[0])=lr_cast(Int32,(((lr_pointer_cast(Int32,pVisitedSize))[0])+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    
    return nothing
end

# Original recordMFFC source device body; candidate, not certified.
@inline function lr_recordMFFC(ctx,vRoots,pFanin0,pFanin1,pNumFanouts,pLevels,vCutTable,vCutSizes,vConeSizes,nPIs,nMaxCutSize,nRoots)::Nothing
    nThreads_L104=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,Int32(gridDim().x)) * lr_cast(UInt32,Int32(blockDim().x))))))
    idx_L105=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local rootId_L106
    local nSaved_L106
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L105) < lr_cast(Int32,nRoots))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vRoots))[lr_cast(Int32,idx_L105)])); rootId_L106 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_findReconvMFFCCut(ctx ,lr_cast(Int32,rootId_L106),lr_pointer_cast(Int32,pFanin0),lr_pointer_cast(Int32,pFanin1),lr_pointer_cast(Int32,pNumFanouts),lr_pointer_cast(Int32,pLevels),lr_pointer_cast(Int32,vCutTable),lr_pointer_cast(Int32,vCutSizes),lr_cast(Int32,nPIs),lr_cast(Int32,nMaxCutSize))); nSaved_L106 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nSaved_L106)); (lr_pointer_cast(Int32,vConeSizes))[lr_cast(Int32,rootId_L106)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(idx_L105 + lr_cast(Int32,nThreads_L104)))); idx_L105 = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original setStatus source device body; candidate, not certified.
@inline function lr_setStatus(ctx,vRoots,vCutTable,vCutSizes,vNodesStatus,nPIs,nRoots)::Nothing
    nThreads_L124=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,Int32(gridDim().x)) * lr_cast(UInt32,Int32(blockDim().x))))))
    idx_L125=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local rootId_L126
    local nodeId_L126
    local nCutSize_L126
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L125) < lr_cast(Int32,nRoots))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vRoots))[lr_cast(Int32,idx_L125)])); rootId_L126 = _lr_rhs; _lr_rhs)
        vCutRoot_L131=lr_pointer_cast(Int32,(lr_pointer_cast(Int32,vCutTable) + lr_cast(Int32,(lr_cast(Int32,rootId_L126) * lr_cast(Int32,lr_cast(Int32,16))))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,rootId_L126)])); nCutSize_L126 = _lr_rhs; _lr_rhs)
        i_L133=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L133) < lr_cast(Int32,nCutSize_L126))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutRoot_L131))[lr_cast(Int32,i_L133)])); nodeId_L126 = _lr_rhs; _lr_rhs)
            if lr_truth((lr_truth(lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,nodeId_L126),lr_cast(Int32,nPIs)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,nodeId_L126)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vNodesStatus))[lr_cast(Int32,nodeId_L126)] = _lr_rhs; _lr_rhs)
            end
            (_lr_old=i_L133; i_L133=lr_cast(Int32,(i_L133+Int32(1))); _lr_old)
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(idx_L125 + lr_cast(Int32,nThreads_L124)))); idx_L125 = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original getCutTruthRanges source device body; candidate, not certified.
@inline function lr_getCutTruthRanges(ctx,vResynRoots,vCutSizes,vCutRanges,vTruthRanges,nResyn)::Nothing
    idx_L144=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L144) < lr_cast(Int32,nResyn))))
        nodeId_L146=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynRoots))[lr_cast(Int32,idx_L144)]))
        cutSize_L147=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutSizes))[lr_cast(Int32,nodeId_L146)]))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,cutSize_L147) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,cutSize_L147)); (lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,idx_L144)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,cutSize_L147))); (lr_pointer_cast(Int32,vTruthRanges))[lr_cast(Int32,idx_L144)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original evalSubgNumAdded source device body; candidate, not certified.
@inline function lr_evalSubgNumAdded(ctx,rootId,pNewRootLevel,vCurrCut,nVars,nNodeMax,nLevelMax,pLevels,vNode2ConeResynIdx,htKeys,htValues,htCapacity,vSubgTable,vSubgLinks,vSubgLens,subgIdx)::Int32
    _lr_mem_0=lr_alloc(Int32)
    lit0_L160=_lr_mem_0
    _lr_mem_1=lr_alloc(Int32)
    lit1_L160=_lr_mem_1
    _lr_mem_2=lr_alloc(Int32)
    fCompRoot_L160=_lr_mem_2
    _lr_mem_3=lr_alloc(LRArray{Int32,256})
    vFuncs_L162=lr_pointer_cast(Int32,_lr_mem_3)
    _lr_mem_4=lr_alloc(LRArray{Int32,256})
    vLevels_L162=lr_pointer_cast(Int32,_lr_mem_4)
    local i_L159
    local counter_L159
    local temp_L159


    local id0_L160
    local id1_L160
    local func0_L160
    local func1_L160
    local currId_L160

    local newLevel_L160
    local retrId_L161


    length_L163=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx)]) + lr_cast(Int32,nVars))))
    local currRowIdx_L164
    local columnPtr_L164
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx)]) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx)]) == lr_cast(Int32,1))))
        lr_factor_unbindAndNodeKeyFlag(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,subgIdx) * lr_cast(Int32,lr_cast(Int32,8))))]),lit0_L160,lit1_L160,fCompRoot_L160)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit0_L160[0]) == lr_cast(Int32,lit1_L160[0]))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); return _lr_result end
        end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L159 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L159) < lr_cast(Int32,nVars))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCurrCut))[lr_cast(Int32,i_L159)])); (lr_pointer_cast(Int32,vFuncs_L162))[lr_cast(Int32,i_L159)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pLevels))[lr_cast(Int32,(lr_pointer_cast(Int32,vCurrCut))[lr_cast(Int32,i_L159)])])); (lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,i_L159)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L159; i_L159=lr_cast(Int32,(i_L159+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); counter_L159 = _lr_rhs; _lr_rhs)
    ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,subgIdx)); currRowIdx_L164 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L164 = _lr_rhs; _lr_rhs))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nVars)); i_L159 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L159) < lr_cast(Int32,length_L163))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,columnPtr_L164) == lr_cast(Int32,lr_cast(Int32,8)))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L164 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L164)])); currRowIdx_L164 = _lr_rhs; _lr_rhs)
        end
        lr_factor_unbindAndNodeKeyFlag(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L164) * lr_cast(Int32,lr_cast(Int32,8)))) + ((_lr_old=columnPtr_L164; columnPtr_L164=lr_cast(Int32,(columnPtr_L164+Int32(1))); _lr_old))))]),lit0_L160,lit1_L160,fCompRoot_L160)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,lit0_L160[0]) < lr_cast(Int32,lit1_L160[0]))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lit0_L160[0]))); id0_L160 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lit1_L160[0]))); id1_L160 = _lr_rhs; _lr_rhs))
        ((lr_truth(lr_cast(Bool,(lr_truth(lr_cast(Bool,(lr_cast(Int32,id0_L160) < lr_cast(Int32,i_L159)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,id1_L160) < lr_cast(Int32,i_L159))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vFuncs_L162))[lr_cast(Int32,id0_L160)])); func0_L160 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vFuncs_L162))[lr_cast(Int32,id1_L160)])); func1_L160 = _lr_rhs; _lr_rhs))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,func0_L160) != lr_cast(Int32,(-lr_cast(Int32,1)))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,func1_L160) != lr_cast(Int32,(-lr_cast(Int32,1))))))))
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,func0_L160),lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L160[0])))); func0_L160 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,func1_L160),lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L160[0])))); func1_L160 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,func0_L160) > lr_cast(Int32,func1_L160))))
                (((_lr_rhs=lr_cast(Int32,lr_cast(Int32,func0_L160)); temp_L159 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,func1_L160)); func0_L160 = _lr_rhs; _lr_rhs)); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L159)); func1_L160 = _lr_rhs; _lr_rhs))
            end
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,func0_L160) <= lr_cast(Int32,func1_L160))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(UInt32,lr_s_retrieveHashTableCheckTrivial(lr_cast(Int32,func0_L160),lr_cast(Int32,func1_L160),lr_pointer_cast(UInt64,htKeys),lr_pointer_cast(UInt32,htValues),lr_cast(Int32,htCapacity))); retrId_L161 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(UInt32,retrId_L161) == lr_cast(UInt32,(typemax(UInt32))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); currId_L160 = _lr_rhs; _lr_rhs)
            else
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,retrId_L161)))); currId_L160 = _lr_rhs; _lr_rhs)
            end
            if lr_truth(lr_cast(Bool,(lr_cast(UInt32,retrId_L161) == lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,rootId))))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); return _lr_result end
            end
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); currId_L160 = _lr_rhs; _lr_rhs)
        end
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,currId_L160) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vNode2ConeResynIdx))[lr_cast(Int32,currId_L160)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(counter_L159=lr_cast(Int32,(counter_L159+Int32(1))); counter_L159)) > lr_cast(Int32,nNodeMax))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); return _lr_result end
            end
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,1) + max(lr_cast(Int32,(lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,id0_L160)]),lr_cast(Int32,(lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,id1_L160)]))))); newLevel_L160 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,currId_L160) != lr_cast(Int32,(-lr_cast(Int32,1))))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,currId_L160) == lr_cast(Int32,0))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); newLevel_L160 = _lr_rhs; _lr_rhs)
            else
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,currId_L160) == lr_s_AigNodeID(lr_cast(Int32,func0_L160)))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,id0_L160)])); newLevel_L160 = _lr_rhs; _lr_rhs)
                else
                    if lr_truth(lr_cast(Bool,(lr_cast(Int32,currId_L160) == lr_s_AigNodeID(lr_cast(Int32,func1_L160)))))
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,id1_L160)])); newLevel_L160 = _lr_rhs; _lr_rhs)
                    end
                end
            end
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,newLevel_L160) > lr_cast(Int32,nLevelMax))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); return _lr_result end
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currId_L160)); (lr_pointer_cast(Int32,vFuncs_L162))[lr_cast(Int32,i_L159)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,newLevel_L160)); (lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,i_L159)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L159; i_L159=lr_cast(Int32,(i_L159+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLevels_L162))[lr_cast(Int32,(lr_cast(Int32,length_L163) - lr_cast(Int32,1)))])); (lr_pointer_cast(Int32,pNewRootLevel))[0] = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,counter_L159)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base)))
    return nothing
end

# Original evalFactoredForm source device body; candidate, not certified.
@inline function lr_evalFactoredForm(ctx,vResynRoots,vCuts,vCutRanges,vNumSaved,pLevels,vNode2ConeResynIdx,htKeys,htValues,htCapacity,vSubgTable,vSubgLinks,vSubgLens,vSelectedSubgInd,nResyn)::Nothing
    _lr_mem_0=lr_alloc(Int32)
    nNewLevel_L277=_lr_mem_0
    nThreads_L274=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,Int32(gridDim().x)) * lr_cast(UInt32,Int32(blockDim().x))))))
    idx_L275=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local coneIdx_L276
    local rootId_L276
    local nVars_L277
    local nSaved_L277
    local nAdded_L277
    local nOtherAdded_L277

    local nOtherNewLevel_L277
    local startIdx_L278
    local endIdx_L278
    local warpMask_L279
    local fSelectedSubg_L280
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L275) < lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,nResyn))))))
        (_lr_rhs=lr_cast(UInt32,CUDA.active_mask()); warpMask_L279 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L275) >> lr_cast(Int32,1)))); coneIdx_L276 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynRoots))[lr_cast(Int32,coneIdx_L276)])); rootId_L276 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,coneIdx_L276) == lr_cast(Int32,0)))) ? lr_cast(Int32,0) : lr_cast(Int32,(lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,(lr_cast(Int32,coneIdx_L276) - lr_cast(Int32,1)))])))); startIdx_L278 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,coneIdx_L276)])); endIdx_L278 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,endIdx_L278) - lr_cast(Int32,startIdx_L278)))); nVars_L277 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vNumSaved))[lr_cast(Int32,rootId_L276)])); nSaved_L277 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_evalSubgNumAdded(ctx ,lr_cast(Int32,rootId_L276),nNewLevel_L277,lr_pointer_cast(Int32,(lr_pointer_cast(Int32,vCuts) + lr_cast(Int32,startIdx_L278))),lr_cast(Int32,nVars_L277),lr_cast(Int32,nSaved_L277),lr_cast(Int32,1000000000),lr_pointer_cast(Int32,pLevels),lr_pointer_cast(Int32,vNode2ConeResynIdx),lr_pointer_cast(UInt64,htKeys),lr_pointer_cast(UInt32,htValues),lr_cast(Int32,htCapacity),lr_pointer_cast(UInt64,vSubgTable),lr_pointer_cast(Int32,vSubgLinks),lr_pointer_cast(Int32,vSubgLens),lr_cast(Int32,idx_L275))); nAdded_L277 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,CUDA.shfl_xor_sync(lr_cast(UInt32,warpMask_L279),lr_cast(Int32,nAdded_L277),lr_cast(Int32,1))); nOtherAdded_L277 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,CUDA.shfl_xor_sync(lr_cast(UInt32,warpMask_L279),lr_cast(Int32,nNewLevel_L277[0]),lr_cast(Int32,1))); nOtherNewLevel_L277 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,rem(lr_cast(Int32,idx_L275),lr_cast(Int32,2))) == lr_cast(Int32,0))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); fSelectedSubg_L280 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nAdded_L277) > lr_cast(Int32,(-lr_cast(Int32,1))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); fSelectedSubg_L280 = _lr_rhs; _lr_rhs)
            end
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nOtherAdded_L277) > lr_cast(Int32,(-lr_cast(Int32,1))))))
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,nAdded_L277) == lr_cast(Int32,(-lr_cast(Int32,1))))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); fSelectedSubg_L280 = _lr_rhs; _lr_rhs)
                else
                    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,nOtherAdded_L277) < lr_cast(Int32,nAdded_L277)))) || lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,nOtherAdded_L277) == lr_cast(Int32,nAdded_L277)))) && lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,(lr_cast(Int32,idx_L275) + lr_cast(Int32,1)))]) == lr_cast(Int32,1)))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,nOtherNewLevel_L277) < lr_cast(Int32,nNewLevel_L277[0]))))))))))))
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); fSelectedSubg_L280 = _lr_rhs; _lr_rhs)
                    end
                end
            end
            (_lr_rhs=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,fSelectedSubg_L280) == lr_cast(Int32,(-lr_cast(Int32,1)))))) ? lr_cast(Int32,(-lr_cast(Int32,1))) : lr_cast(Int32,(lr_cast(Int32,idx_L275) + lr_cast(Int32,fSelectedSubg_L280)))))); (lr_pointer_cast(Int32,vSelectedSubgInd))[lr_cast(Int32,coneIdx_L276)] = _lr_rhs; _lr_rhs)
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(idx_L275 + lr_cast(Int32,nThreads_L274)))); idx_L275 = _lr_rhs; _lr_rhs)
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end

# Original duplicateHashTableWithoutMFFCs source device body; candidate, not certified.
@inline function lr_duplicateHashTableWithoutMFFCs(ctx,vNode2ConeResynIdx,vSelectedSubgInd,htKeys,htValues,htCapacity,htDestKeys,htDestValues,htDestCapacity)::Nothing
    idx_L323=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L323) < lr_cast(Int32,htCapacity)))) && lr_truth(lr_cast(Bool,(lr_cast(UInt64,(lr_pointer_cast(UInt64,htKeys))[lr_cast(Int32,idx_L323)]) != lr_cast(UInt64,lr_cast(UInt64,-1)))))))
        nodeId_L325=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(UInt32,htValues))[lr_cast(Int32,idx_L323)])))
        coneResynIdx_L326=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vNode2ConeResynIdx))[lr_cast(Int32,nodeId_L325)]))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,coneResynIdx_L326) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSelectedSubgInd))[lr_cast(Int32,coneResynIdx_L326)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))))
            lr_s_insert_single_no_update(lr_pointer_cast(UInt64,htDestKeys),lr_pointer_cast(UInt32,htDestValues),lr_cast(UInt64,(lr_pointer_cast(UInt64,htKeys))[lr_cast(Int32,idx_L323)]),lr_cast(UInt32,(lr_pointer_cast(UInt32,htValues))[lr_cast(Int32,idx_L323)]),lr_cast(Int32,htDestCapacity))
        end
    end
    
    return nothing
end

# Original checkSingleVarSubg source device body; candidate, not certified.
@inline function lr_checkSingleVarSubg(ctx,vSubgTable,vSubgLinks,vSubgLens,vResynRoots,vSelectedSubgInd,vOldRoot2NewRootLits,vFinishedMark,nObjs,nResyn)::Nothing
    _lr_mem_0=lr_alloc(Int32)
    lit0_L344=_lr_mem_0
    _lr_mem_1=lr_alloc(Int32)
    lit1_L344=_lr_mem_1
    _lr_mem_2=lr_alloc(Int32)
    fComp_L344=_lr_mem_2
    idx_L340=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L340) < lr_cast(Int32,nResyn))))
        subgIdx_L342=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSelectedSubgInd))[lr_cast(Int32,idx_L340)]))
        rootId_L343=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynRoots))[lr_cast(Int32,idx_L340)]))



        if lr_truth(lr_cast(Bool,(lr_cast(Int32,subgIdx_L342) == lr_cast(Int32,(-lr_cast(Int32,1))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vFinishedMark))[lr_cast(Int32,idx_L340)] = _lr_rhs; _lr_rhs)
        else
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx_L342)]) == lr_cast(Int32,1))))
                lr_factor_unbindAndNodeKeyFlag(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,subgIdx_L342) * lr_cast(Int32,lr_cast(Int32,8))))]),lit0_L344,lit1_L344,fComp_L344)
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit0_L344[0]) == lr_cast(Int32,lit1_L344[0]))))
                    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_s_AigNodeID(lr_cast(Int32,lit0_L344[0])) < lr_cast(Int32,nObjs))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
                    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,fComp_L344[0]) == lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L344[0])))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit0_L344[0])); (lr_pointer_cast(Int32,vOldRoot2NewRootLits))[lr_cast(Int32,rootId_L343)] = _lr_rhs; _lr_rhs)
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vFinishedMark))[lr_cast(Int32,idx_L340)] = _lr_rhs; _lr_rhs)
                end
            end
        end
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base)))
    return nothing
end

# Original insertSubgIter source device body; candidate, not certified.
@inline function lr_insertSubgIter(ctx,iter,vResynIdSeq,vCuts,vCutRanges,htDestKeys,htDestValues,htDestCapacity,vSubgTable,vSubgLinks,vSubgLens,vSelectedSubgInd,idCounter,nReplace)::Nothing
    _lr_mem_0=lr_alloc(Int32)
    lit0_L372=_lr_mem_0
    _lr_mem_1=lr_alloc(Int32)
    lit1_L372=_lr_mem_1
    _lr_mem_2=lr_alloc(Int32)
    fComp_L372=_lr_mem_2
    _lr_mem_3=lr_alloc(UInt32)
    temp0_L373=_lr_mem_3
    _lr_mem_4=lr_alloc(UInt32)
    temp1_L373=_lr_mem_4
    _lr_mem_5=lr_alloc(LRArray{Int32,32})
    subgRows_L376=lr_pointer_cast(Int32,_lr_mem_5)
    idx_L369=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L369) < lr_cast(Int32,nReplace))))
        local startIdx_L371
        local endIdx_L371
        local nVars_L371


        local id0_L372
        local id1_L372
        local fanin0_L372
        local fanin1_L372

        local temp_L372


        local key_L374
        local currRowIdx_L375
        local columnPtr_L375

        resynIdx_L377=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynIdSeq))[lr_cast(Int32,idx_L369)]))
        subgIdx_L378=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSelectedSubgInd))[lr_cast(Int32,resynIdx_L377)]))
        (_lr_rhs=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,resynIdx_L377) == lr_cast(Int32,0)))) ? lr_cast(Int32,0) : lr_cast(Int32,(lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,(lr_cast(Int32,resynIdx_L377) - lr_cast(Int32,1)))])))); startIdx_L371 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,resynIdx_L377)])); endIdx_L371 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,endIdx_L371) - lr_cast(Int32,startIdx_L371)))); nVars_L371 = _lr_rhs; _lr_rhs)
        vCurrCut_L383=lr_pointer_cast(Int32,lr_pointer_cast(Int32,(lr_pointer_cast(Int32,vCuts) + lr_cast(Int32,startIdx_L371))))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iter) < lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx_L378)]))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iter) < lr_cast(Int32,lr_cast(Int32,256)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,subgIdx_L378)); currRowIdx_L375 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rem(lr_cast(Int32,iter),lr_cast(Int32,lr_cast(Int32,8))))); columnPtr_L375 = _lr_rhs; _lr_rhs))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currRowIdx_L375)); (lr_pointer_cast(Int32,subgRows_L376))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
        i_L390=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L390) < (lr_cast(Int32,div(lr_cast(Int32,iter),lr_cast(Int32,lr_cast(Int32,8))))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L375)])); currRowIdx_L375 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currRowIdx_L375)); (lr_pointer_cast(Int32,subgRows_L376))[lr_cast(Int32,(lr_cast(Int32,i_L390) + lr_cast(Int32,1)))] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L390; i_L390=lr_cast(Int32,(i_L390+Int32(1))); _lr_old)
        end
        lr_factor_unbindAndNodeKeyFlag(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L375) * lr_cast(Int32,lr_cast(Int32,8)))) + lr_cast(Int32,columnPtr_L375)))]),lit0_L372,lit1_L372,fComp_L372)
        ((_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lit0_L372[0]))); id0_L372 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lit1_L372[0]))); id1_L372 = _lr_rhs; _lr_rhs))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,id0_L372) < lr_cast(Int32,nVars_L371))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCurrCut_L383))[lr_cast(Int32,id0_L372)])); fanin0_L372 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,fanin0_L372),lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L372[0])))); fanin0_L372 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(id0_L372 - lr_cast(Int32,nVars_L371)))); id0_L372 = _lr_rhs; _lr_rhs)
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,id0_L372) < lr_cast(Int32,iter))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,subgRows_L376))[lr_cast(Int32,div(lr_cast(Int32,id0_L372),lr_cast(Int32,lr_cast(Int32,8))))])); currRowIdx_L375 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rem(lr_cast(Int32,id0_L372),lr_cast(Int32,lr_cast(Int32,8))))); columnPtr_L375 = _lr_rhs; _lr_rhs))
            lr_s_unbindAndNodeKeys(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L375) * lr_cast(Int32,lr_cast(Int32,8)))) + lr_cast(Int32,columnPtr_L375)))]),temp0_L373,temp1_L373)
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,temp0_L373[0]) == lr_cast(UInt32,lr_cast(Int32,0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,temp1_L373[0])))); fanin0_L372 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeNotCond(lr_cast(Int32,fanin0_L372),lr_s_AigNodeIsComplement(lr_cast(Int32,lit0_L372[0])))); fanin0_L372 = _lr_rhs; _lr_rhs)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,id1_L372) < lr_cast(Int32,nVars_L371))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCurrCut_L383))[lr_cast(Int32,id1_L372)])); fanin1_L372 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,fanin1_L372),lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L372[0])))); fanin1_L372 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(id1_L372 - lr_cast(Int32,nVars_L371)))); id1_L372 = _lr_rhs; _lr_rhs)
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,id1_L372) < lr_cast(Int32,iter))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,subgRows_L376))[lr_cast(Int32,div(lr_cast(Int32,id1_L372),lr_cast(Int32,lr_cast(Int32,8))))])); currRowIdx_L375 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rem(lr_cast(Int32,id1_L372),lr_cast(Int32,lr_cast(Int32,8))))); columnPtr_L375 = _lr_rhs; _lr_rhs))
            lr_s_unbindAndNodeKeys(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L375) * lr_cast(Int32,lr_cast(Int32,8)))) + lr_cast(Int32,columnPtr_L375)))]),temp0_L373,temp1_L373)
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,temp0_L373[0]) == lr_cast(UInt32,lr_cast(Int32,0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,temp1_L373[0])))); fanin1_L372 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeNotCond(lr_cast(Int32,fanin1_L372),lr_s_AigNodeIsComplement(lr_cast(Int32,lit1_L372[0])))); fanin1_L372 = _lr_rhs; _lr_rhs)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,fanin0_L372) > lr_cast(Int32,fanin1_L372))))
            (((_lr_rhs=lr_cast(Int32,lr_cast(Int32,fanin0_L372)); temp_L372 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,fanin1_L372)); fanin0_L372 = _lr_rhs; _lr_rhs)); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L372)); fanin1_L372 = _lr_rhs; _lr_rhs))
        end
        (_lr_rhs=lr_cast(UInt32,lr_s_checkTrivialAndCases(lr_cast(Int32,fanin0_L372),lr_cast(Int32,fanin1_L372))); temp0_L373[0] = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,temp0_L373[0]) == lr_cast(UInt32,typemax(UInt32)))))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,fanin0_L372) < lr_cast(Int32,fanin1_L372))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,fanin0_L372),lr_cast(Int32,fanin1_L372))); key_L374 = _lr_rhs; _lr_rhs)
            lr_s_insert_single_no_update(lr_pointer_cast(UInt64,htDestKeys),lr_pointer_cast(UInt32,htDestValues),lr_cast(UInt64,key_L374),lr_cast(UInt32,lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,idCounter) + lr_cast(Int32,idx_L369)))))),lr_cast(Int32,htDestCapacity))
        end
        (_lr_rhs=lr_cast(UInt64,lr_factor_formAndNodeKeyFlag(lr_cast(Int32,fanin0_L372),lr_cast(Int32,fanin1_L372),lr_cast(Int32,fComp_L372[0]))); key_L374 = _lr_rhs; _lr_rhs)
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,subgRows_L376))[lr_cast(Int32,div(lr_cast(Int32,iter),lr_cast(Int32,lr_cast(Int32,8))))])); currRowIdx_L375 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rem(lr_cast(Int32,iter),lr_cast(Int32,lr_cast(Int32,8))))); columnPtr_L375 = _lr_rhs; _lr_rhs))
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,key_L374)); (lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L375) * lr_cast(Int32,lr_cast(Int32,8)))) + lr_cast(Int32,columnPtr_L375)))] = _lr_rhs; _lr_rhs)
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_5,:base)))
    return nothing
end

# Original updateInsertedIdsIter source device body; candidate, not certified.
@inline function lr_updateInsertedIdsIter(ctx,iter,vResynRoots,vResynIdSeq,htDestKeys,htDestValues,htDestCapacity,vSubgTable,vSubgLinks,vSubgLens,vSelectedSubgInd,vOldRoot2NewRootLits,vFinishedMark,nReplace)::Nothing
    _lr_mem_0=lr_alloc(Int32)
    fanin0_L450=_lr_mem_0
    _lr_mem_1=lr_alloc(Int32)
    fanin1_L450=_lr_mem_1
    _lr_mem_2=lr_alloc(Int32)
    fComp_L450=_lr_mem_2
    idx_L448=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L448) < lr_cast(Int32,nReplace))))



        local temp0_L451
        local key_L452
        local currRowIdx_L453
        local columnPtr_L453
        resynIdx_L454=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynIdSeq))[lr_cast(Int32,idx_L448)]))
        subgIdx_L455=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSelectedSubgInd))[lr_cast(Int32,resynIdx_L454)]))
        oldRootId_L456=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynRoots))[lr_cast(Int32,resynIdx_L454)]))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iter) < lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx_L455)]))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iter) < lr_cast(Int32,lr_cast(Int32,256)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,subgIdx_L455)); currRowIdx_L453 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rem(lr_cast(Int32,iter),lr_cast(Int32,lr_cast(Int32,8))))); columnPtr_L453 = _lr_rhs; _lr_rhs))
        i_L462=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L462) < (lr_cast(Int32,div(lr_cast(Int32,iter),lr_cast(Int32,lr_cast(Int32,8))))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L453)])); currRowIdx_L453 = _lr_rhs; _lr_rhs)
            (_lr_old=i_L462; i_L462=lr_cast(Int32,(i_L462+Int32(1))); _lr_old)
        end
        lr_factor_unbindAndNodeKeyFlag(lr_cast(UInt64,(lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L453) * lr_cast(Int32,lr_cast(Int32,8)))) + lr_cast(Int32,columnPtr_L453)))]),fanin0_L450,fanin1_L450,fComp_L450)
        (_lr_rhs=lr_cast(UInt32,lr_s_checkTrivialAndCases(lr_cast(Int32,fanin0_L450[0]),lr_cast(Int32,fanin1_L450[0]))); temp0_L451 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,temp0_L451) == lr_cast(UInt32,typemax(UInt32)))))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,fanin0_L450[0]) < lr_cast(Int32,fanin1_L450[0]))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,fanin0_L450[0]),lr_cast(Int32,fanin1_L450[0]))); key_L452 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(UInt32,lr_s_retrieve_single(lr_pointer_cast(UInt64,htDestKeys),lr_pointer_cast(UInt32,htDestValues),lr_cast(UInt64,key_L452),lr_cast(Int32,htDestCapacity))); temp0_L451 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,temp0_L451) << lr_cast(Int32,1)))); temp0_L451 = _lr_rhs; _lr_rhs)
        end
        (_lr_rhs=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,0),lr_cast(Int32,lr_cast(UInt32,temp0_L451)))); key_L452 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,key_L452)); (lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L453) * lr_cast(Int32,lr_cast(Int32,8)))) + lr_cast(Int32,columnPtr_L453)))] = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,iter) == lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,subgIdx_L455)]) - lr_cast(Int32,1))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vFinishedMark))[lr_cast(Int32,idx_L448)] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeNotCond(lr_cast(Int32,lr_cast(UInt32,temp0_L451)),lr_cast(Int32,fComp_L450[0]))); (lr_pointer_cast(Int32,vOldRoot2NewRootLits))[lr_cast(Int32,oldRootId_L456)] = _lr_rhs; _lr_rhs)
        end
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base)))
    return nothing
end

# Original checkInsertion source device body; candidate, not certified.
@inline function lr_checkInsertion(ctx,vResynRoots,vOldRoot2NewRootLits,vSelectedSubgInd,nResyn)::Nothing
    idx_L492=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L492) < lr_cast(Int32,nResyn))))
        subgIdx_L494=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vSelectedSubgInd))[lr_cast(Int32,idx_L492)]))
        rootId_L495=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vResynRoots))[lr_cast(Int32,idx_L492)]))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,subgIdx_L494) != lr_cast(Int32,(-lr_cast(Int32,1))))))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vOldRoot2NewRootLits))[lr_cast(Int32,rootId_L495)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        end
    end
    
    return nothing
end

# Original unbindKeysUpdateOldRoots source device body; candidate, not certified.
@inline function lr_unbindKeysUpdateOldRoots(ctx,vOldRoot2NewRootLits,vReconstructedKeys,vReconstructedIds,vFanin0New,vFanin1New,nEntries,nObjs,nBufferLen)::Nothing
    _lr_mem_0=lr_alloc(UInt32)
    lit0_L507=_lr_mem_0
    _lr_mem_1=lr_alloc(UInt32)
    lit1_L507=_lr_mem_1
    idx_L505=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L505) < lr_cast(Int32,nEntries))))


        local nodeId_L507
        local newLit0_L508
        local newLit1_L508
        lr_s_unbindAndNodeKeys(lr_cast(UInt64,(lr_pointer_cast(UInt64,vReconstructedKeys))[lr_cast(Int32,idx_L505)]),lit0_L507,lit1_L507)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,vReconstructedIds))[lr_cast(Int32,idx_L505)])); nodeId_L507 = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,nodeId_L507) < lr_cast(UInt32,lr_cast(Int32,nBufferLen)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(UInt32,nodeId_L507) < lr_cast(UInt32,lr_cast(Int32,nObjs))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vOldRoot2NewRootLits))[lr_cast(UInt32,nodeId_L507)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,0))); newLit0_L508 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vOldRoot2NewRootLits))[lr_cast(UInt32,nodeId_L507)])); newLit1_L508 = _lr_rhs; _lr_rhs)
        else
            ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,lit0_L507[0])))); newLit0_L508 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,lit1_L507[0])))); newLit1_L508 = _lr_rhs; _lr_rhs))
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,newLit0_L508)); (lr_pointer_cast(Int32,vFanin0New))[lr_cast(UInt32,nodeId_L507)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,newLit1_L508)); (lr_pointer_cast(Int32,vFanin1New))[lr_cast(UInt32,nodeId_L507)] = _lr_rhs; _lr_rhs)
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original getElemTruthTable source device body; candidate, not certified.
@inline function lr_getElemTruthTable(ctx,vTruthElem,nVars)::Nothing
    _lr_mem_0=lr_alloc(LRArray{UInt32,5})
    masks_L536=lr_pointer_cast(UInt32,_lr_mem_0)
    idx_L535=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    LRPtr{UInt32}(getfield(masks_L536,:base))[0]=lr_cast(UInt32,2863311530)
    LRPtr{UInt32}(getfield(masks_L536,:base))[1]=lr_cast(UInt32,3435973836)
    LRPtr{UInt32}(getfield(masks_L536,:base))[2]=lr_cast(UInt32,4042322160)
    LRPtr{UInt32}(getfield(masks_L536,:base))[3]=lr_cast(UInt32,4278255360)
    LRPtr{UInt32}(getfield(masks_L536,:base))[4]=lr_cast(UInt32,4294901760)
    local pTruth_L537
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L535) == lr_cast(Int32,0))))
        nWords_L540=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVars)))
        i_L542=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L542) < lr_cast(Int32,nVars))))
            (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,vTruthElem) + lr_cast(Int32,(lr_cast(Int32,i_L542) * lr_cast(Int32,nWords_L540)))))); pTruth_L537 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L542) < lr_cast(Int32,5))))
                k_L545=lr_cast(Int32,lr_cast(Int32,0))
                while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L545) < lr_cast(Int32,nWords_L540))))
                    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,masks_L536))[lr_cast(Int32,i_L542)])); (lr_pointer_cast(UInt32,pTruth_L537))[lr_cast(Int32,k_L545)] = _lr_rhs; _lr_rhs)
                    (_lr_old=k_L545; k_L545=lr_cast(Int32,(k_L545+Int32(1))); _lr_old)
                end
            else
                k_L548=lr_cast(Int32,lr_cast(Int32,0))
                while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L548) < lr_cast(Int32,nWords_L540))))
                    if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,k_L548) & (lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,i_L542) - lr_cast(Int32,5)))))))))))
                        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(~lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0)))))); (lr_pointer_cast(UInt32,pTruth_L537))[lr_cast(Int32,k_L548)] = _lr_rhs; _lr_rhs)
                    else
                        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); (lr_pointer_cast(UInt32,pTruth_L537))[lr_cast(Int32,k_L548)] = _lr_rhs; _lr_rhs)
                    end
                    (_lr_old=k_L548; k_L548=lr_cast(Int32,(k_L548+Int32(1))); _lr_old)
                end
            end
            (_lr_old=i_L542; i_L542=lr_cast(Int32,(i_L542+Int32(1))); _lr_old)
        end
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end
