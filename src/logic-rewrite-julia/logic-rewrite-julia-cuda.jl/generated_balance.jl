# GENERATED from original source; full-pipeline verification PENDING.
# Original checkCoverTravEnd source device body; candidate, not certified.
@inline function lr_checkCoverTravEnd(ctx,isEnd,vInputs,vInputsFilter,vCanonTable,nPIs,arrayLen)::Nothing
    idx_L75=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L75) < lr_cast(Int32,arrayLen))))
        thisLit_L77=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vInputs))[lr_cast(Int32,idx_L75)]))
        nodeId_L78=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,thisLit_L77)))
        nodeIsComplement_L79=lr_cast(Int32,lr_s_AigNodeIsComplement(lr_cast(Int32,thisLit_L77)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeId_L78)); (lr_pointer_cast(Int32,vInputs))[lr_cast(Int32,idx_L75)] = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,nodeId_L78),lr_cast(Int32,nPIs))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,isEnd))[0] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vInputsFilter))[lr_cast(Int32,idx_L75)] = _lr_rhs; _lr_rhs)
            old_L93=lr_cast(Int32,lr_atomicCAS((lr_pointer_cast(Int32,vCanonTable) + lr_cast(Int32,nodeId_L78)),lr_cast(Int32,0),lr_cast(Int32,(lr_cast(Int32,idx_L75) + lr_cast(Int32,1)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,old_L93) > lr_cast(Int32,0))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,old_L93)))); (lr_pointer_cast(Int32,vInputsFilter))[lr_cast(Int32,idx_L75)] = _lr_rhs; _lr_rhs)
            end
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vInputsFilter))[lr_cast(Int32,idx_L75)] = _lr_rhs; _lr_rhs)
        end
    end
    
    return nothing
end

# Original retrieveAndNode source device body; candidate, not certified.
@inline function lr_retrieveAndNode(ctx,lit1,lit2,htReconstructKeys,htReconstructValues,htReconstructCapacity)::UInt32
    trivialResult_L120=lr_cast(UInt32,lr_s_checkTrivialAndCases(lr_cast(Int32,lit1),lr_cast(Int32,lit2)))
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,trivialResult_L120) != lr_cast(UInt32,typemax(UInt32)))))
        begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,trivialResult_L120)); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit1) > lr_cast(Int32,lit2))))
        local temp_L126
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit1)); temp_L126 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit2)); lit1 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L126)); lit2 = _lr_rhs; _lr_rhs)
    end
    key_L131=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,lit1),lr_cast(Int32,lit2)))
    value_L134=lr_cast(UInt32,lr_s_retrieve_single(lr_pointer_cast(UInt64,htReconstructKeys),lr_pointer_cast(UInt32,htReconstructValues),lr_cast(UInt64,key_L131),lr_cast(Int32,htReconstructCapacity)))
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,value_L134) != lr_cast(UInt32,typemax(UInt32)))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,value_L134) << lr_cast(Int32,1)))); value_L134 = _lr_rhs; _lr_rhs)
    end
    begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,value_L134)); ; return _lr_result end
    
    return nothing
end

# Original prepareDataToInsert source device body; candidate, not certified.
@inline function lr_prepareDataToInsert(ctx,vLocalReconstructArrays,vLocalReconstructLevels,vLocalReconstructLens,vStepNodeKeys,vStepNodeValues,vStepNodeMask,vStepNodeLevels,htReconstructKeys,htReconstructValues,htReconstructCapacity,htLevelKeys,htLevelValues,htLevelCapacity,newIdCounter,nReadyCovers,nPIs,nPOs,maxCoverLen)::Nothing
    idx_L158=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L158) < lr_cast(Int32,nReadyCovers))))
        length_L160=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L158)]))
        localArrayStartIdx_L161=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L158) * lr_cast(Int32,maxCoverLen))))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,length_L160) >= lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,length_L160) <= lr_cast(Int32,1))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vStepNodeMask))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
            begin _lr_result=(nothing; nothing); ; return _lr_result end
        end
        lit1_L171=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L161) + lr_cast(Int32,length_L160))) - lr_cast(Int32,1)))]))
        lit2_L172=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L161) + lr_cast(Int32,length_L160))) - lr_cast(Int32,2)))]))
        lv1_L173=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L161) + lr_cast(Int32,length_L160))) - lr_cast(Int32,1)))]))
        lv2_L174=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L161) + lr_cast(Int32,length_L160))) - lr_cast(Int32,2)))]))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L158)] - lr_cast(Int32,2)))); (lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
        retrieveRes_L179=lr_cast(UInt32,lr_retrieveAndNode(ctx ,lr_cast(Int32,lit1_L171),lr_cast(Int32,lit2_L172),lr_pointer_cast(UInt64,htReconstructKeys),lr_pointer_cast(UInt32,htReconstructValues),lr_cast(Int32,htReconstructCapacity)))
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,retrieveRes_L179) != lr_cast(UInt32,typemax(UInt32)))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,2)); (lr_pointer_cast(Int32,vStepNodeMask))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,retrieveRes_L179)); (lr_pointer_cast(UInt32,vStepNodeValues))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
            retrievedId_L189=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,lr_cast(UInt32,retrieveRes_L179))))
            if lr_truth(lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,retrievedId_L189),lr_cast(Int32,nPIs))))
                (_lr_rhs=lr_cast(UInt32,lr_s_retrieve_single(lr_pointer_cast(UInt32,htLevelKeys),lr_pointer_cast(UInt32,htLevelValues),lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,retrievedId_L189))),lr_cast(Int32,htLevelCapacity))); retrieveRes_L179 = _lr_rhs; _lr_rhs)
            else
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); retrieveRes_L179 = _lr_rhs; _lr_rhs)
            end
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,retrieveRes_L179) != lr_cast(UInt32,(typemax(UInt32))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,retrieveRes_L179)); (lr_pointer_cast(UInt32,vStepNodeLevels))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
            begin _lr_result=(nothing; nothing); ; return _lr_result end
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit1_L171) > lr_cast(Int32,lit2_L172))))
            local temp_L205
            (((_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit1_L171)); temp_L205 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit2_L172)); lit1_L171 = _lr_rhs; _lr_rhs)); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L205)); lit2_L172 = _lr_rhs; _lr_rhs))
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vStepNodeMask))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_s_formAndNodeKey(lr_cast(Int32,lit1_L171),lr_cast(Int32,lit2_L172))); (lr_pointer_cast(UInt64,vStepNodeKeys))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,newIdCounter) + lr_cast(Int32,idx_L158))))))); (lr_pointer_cast(UInt32,vStepNodeValues))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,lr_cast(UInt32,max(lr_cast(Int32,lv1_L173),lr_cast(Int32,lv2_L174)))) + lr_cast(UInt32,lr_cast(Int32,1))))); (lr_pointer_cast(UInt32,vStepNodeLevels))[lr_cast(Int32,idx_L158)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original sharedNodeDrivenPermute source device body; candidate, not certified.
@inline function lr_sharedNodeDrivenPermute(ctx,vLocalReconstructArrays,vLocalReconstructLevels,vLocalReconstructLens,htReconstructKeys,htReconstructValues,htReconstructCapacity,nReadyCovers,maxCoverLen)::Nothing
    idx_L222=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L222) < lr_cast(Int32,nReadyCovers))))
        local leftMostIdx_L224
        local rightBoundIdx_L224
        length_L225=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L222)]))
        localArrayStartIdx_L226=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L222) * lr_cast(Int32,maxCoverLen))))
        local borderLevel_L227
        local currLevel_L227
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,length_L225) <= lr_cast(Int32,1))))
            begin _lr_result=(nothing; nothing); ; return _lr_result end
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,length_L225) < lr_cast(Int32,3))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); leftMostIdx_L224 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,length_L225) - lr_cast(Int32,2)))); leftMostIdx_L224 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,leftMostIdx_L224)))])); borderLevel_L227 = _lr_rhs; _lr_rhs)
            (_lr_old=leftMostIdx_L224; leftMostIdx_L224=lr_cast(Int32,(leftMostIdx_L224-Int32(1))); _lr_old)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,leftMostIdx_L224) >= lr_cast(Int32,0))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,leftMostIdx_L224)))])); currLevel_L227 = _lr_rhs; _lr_rhs)
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,currLevel_L227) != lr_cast(Int32,borderLevel_L227))))
                    break
                end
                (_lr_old=leftMostIdx_L224; leftMostIdx_L224=lr_cast(Int32,(leftMostIdx_L224-Int32(1))); _lr_old)
            end
            (_lr_old=leftMostIdx_L224; leftMostIdx_L224=lr_cast(Int32,(leftMostIdx_L224+Int32(1))); _lr_old)
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,leftMostIdx_L224)))]) == lr_cast(Int32,borderLevel_L227))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,length_L225) - lr_cast(Int32,2)))); rightBoundIdx_L224 = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,leftMostIdx_L224) <= lr_cast(Int32,rightBoundIdx_L224))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,leftMostIdx_L224) < lr_cast(Int32,rightBoundIdx_L224))))
            local i_L253
            local lit1_L253
            local lit2_L253
            local lit3_L253
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,rightBoundIdx_L224))) + lr_cast(Int32,1)))])); lit1_L253 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,rightBoundIdx_L224)))])); lit2_L253 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rightBoundIdx_L224)); i_L253 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L253) >= lr_cast(Int32,leftMostIdx_L224))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,i_L253)))])); lit3_L253 = _lr_rhs; _lr_rhs)
                if lr_truth(lr_cast(Bool,(lr_retrieveAndNode(ctx ,lr_cast(Int32,lit1_L253),lr_cast(Int32,lit3_L253),lr_pointer_cast(UInt64,htReconstructKeys),lr_pointer_cast(UInt32,htReconstructValues),lr_cast(Int32,htReconstructCapacity)) != lr_cast(UInt32,typemax(UInt32)))))
                    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lit3_L253) != lr_cast(Int32,lit2_L253))))
                        local tempLevel_L264
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit2_L253)); (lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,i_L253)))] = _lr_rhs; _lr_rhs)
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lit3_L253)); (lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,rightBoundIdx_L224)))] = _lr_rhs; _lr_rhs)
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,i_L253)))])); tempLevel_L264 = _lr_rhs; _lr_rhs)
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,rightBoundIdx_L224)))])); (lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,i_L253)))] = _lr_rhs; _lr_rhs)
                        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,tempLevel_L264)); (lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L226) + lr_cast(Int32,rightBoundIdx_L224)))] = _lr_rhs; _lr_rhs)
                    end
                    break
                end
                (_lr_old=i_L253; i_L253=lr_cast(Int32,(i_L253-Int32(1))); _lr_old)
            end
        end
    end
    
    return nothing
end

# Original addBackLocalArrays source device body; candidate, not certified.
@inline function lr_addBackLocalArrays(ctx,vLocalReconstructArrays,vLocalReconstructLevels,vLocalReconstructLens,vStepNodeValues,vStepNodeMask,vStepNodeLevels,nReadyCovers,maxCoverLen)::Nothing
    idx_L462=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L462) < lr_cast(Int32,nReadyCovers)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vStepNodeMask))[lr_cast(Int32,idx_L462)]) != lr_cast(Int32,0))))))
        litToInsert_L465=lr_cast(Int32,lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vStepNodeMask))[lr_cast(Int32,idx_L462)]) == lr_cast(Int32,2)))) ? lr_cast(UInt32,(lr_pointer_cast(UInt32,vStepNodeValues))[lr_cast(Int32,idx_L462)]) : (lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,vStepNodeValues))[lr_cast(Int32,idx_L462)]) << lr_cast(Int32,1))))))))
        localArrayStartIdx_L466=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L462) * lr_cast(Int32,maxCoverLen))))
        length_L467=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L462)]))
        local i_L468
        local temp_L469
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L468 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L468) < lr_cast(Int32,length_L467))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468)))]) == lr_cast(Int32,litToInsert_L465))))
                begin _lr_result=(nothing; nothing); ; return _lr_result end
            end
            (_lr_old=i_L468; i_L468=lr_cast(Int32,(i_L468+Int32(1))); _lr_old)
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,litToInsert_L465)); (lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,length_L467)))] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(UInt32,vStepNodeLevels))[lr_cast(Int32,idx_L462)]))); (lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,length_L467)))] = _lr_rhs; _lr_rhs)
        (_lr_old=(lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L462)]; (lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L462)]=lr_cast(Int32,((lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L462)]+Int32(1))); _lr_old)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,length_L467)); i_L468 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L468) > lr_cast(Int32,0))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468)))]) <= lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468))) - lr_cast(Int32,1)))]))))
                break
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468)))])); temp_L469 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468))) - lr_cast(Int32,1)))])); (lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468)))] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L469)); (lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468))) - lr_cast(Int32,1)))] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468)))])); temp_L469 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468))) - lr_cast(Int32,1)))])); (lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468)))] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L469)); (lr_pointer_cast(Int32,vLocalReconstructLevels))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,localArrayStartIdx_L466) + lr_cast(Int32,i_L468))) - lr_cast(Int32,1)))] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L468; i_L468=lr_cast(Int32,(i_L468-Int32(1))); _lr_old)
        end
    end
    
    return nothing
end

# Original recordReconstructedCovers source device body; candidate, not certified.
@inline function lr_recordReconstructedCovers(ctx,vLocalReconstructArrays,vLocalReconstructLens,vGatheredReadyCovers,vNodeCoverIdMapping,vCoverNodeNewLitMapping,nReadyCovers,maxCoverLen)::Nothing
    idx_L502=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L502) < lr_cast(Int32,nReadyCovers))))
        local newLit_L504
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L502)]) == lr_cast(Int32,0))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,1))); newLit_L504 = _lr_rhs; _lr_rhs)
        else
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructLens))[lr_cast(Int32,idx_L502)]) == lr_cast(Int32,1))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vLocalReconstructArrays))[lr_cast(Int32,(lr_cast(Int32,idx_L502) * lr_cast(Int32,maxCoverLen)))])); newLit_L504 = _lr_rhs; _lr_rhs)
        end
        nodeId_L517=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vGatheredReadyCovers))[lr_cast(Int32,idx_L502)]))
        coverId_L518=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vNodeCoverIdMapping))[lr_cast(Int32,nodeId_L517)]))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,newLit_L504)); (lr_pointer_cast(Int32,vCoverNodeNewLitMapping))[lr_cast(Int32,coverId_L518)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original genReadyMask source device body; candidate, not certified.
@inline function lr_genReadyMask(ctx,vCoverNodeIdMapping,vNodeCoverIdMapping,vCoverTable,vCoverTableLinks,vCoverTableLens,vCoverNodeNewLitMapping,vThisIterReady,arrayLen,nPIs)::Nothing
    idx_L528=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L528) > lr_cast(Int32,0)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L528) < lr_cast(Int32,arrayLen))))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverNodeNewLitMapping))[lr_cast(Int32,idx_L528)]) >= lr_cast(Int32,0))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vThisIterReady))[lr_cast(Int32,idx_L528)] = _lr_rhs; _lr_rhs)
        else
            nodeId_L536=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverNodeIdMapping))[lr_cast(Int32,idx_L528)]))
            ready_L537=lr_cast(Int32,lr_cast(Int32,1))
            currRowIdx_L538=lr_cast(Int32,lr_cast(Int32,nodeId_L536))
            columnPtr_L539=lr_cast(Int32,lr_cast(Int32,0))
            length_L540=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLens))[lr_cast(Int32,nodeId_L536)]))
            local currInputId_L541
            i_L543=lr_cast(Int32,lr_cast(Int32,0))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L543) < lr_cast(Int32,length_L540))))
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,columnPtr_L539) == lr_cast(Int32,lr_cast(Int32,4)))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L539 = _lr_rhs; _lr_rhs)
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,currRowIdx_L538)])); currRowIdx_L538 = _lr_rhs; _lr_rhs)
                end
                (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L538) * lr_cast(Int32,lr_cast(Int32,4)))) + ((_lr_old=columnPtr_L539; columnPtr_L539=lr_cast(Int32,(columnPtr_L539+Int32(1))); _lr_old))))]))); currInputId_L541 = _lr_rhs; _lr_rhs)
                if lr_truth(lr_cast(Bool,lr_s_AigIsPIConst(lr_cast(Int32,currInputId_L541),lr_cast(Int32,nPIs))))
                    (_lr_old=i_L543; i_L543=lr_cast(Int32,(i_L543+Int32(1))); _lr_old); continue
                end
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverNodeNewLitMapping))[lr_cast(Int32,(lr_pointer_cast(Int32,vNodeCoverIdMapping))[lr_cast(Int32,currInputId_L541)])]) == lr_cast(Int32,(-lr_cast(Int32,1))))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); ready_L537 = _lr_rhs; _lr_rhs)
                    break
                end
                (_lr_old=i_L543; i_L543=lr_cast(Int32,(i_L543+Int32(1))); _lr_old)
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,ready_L537)); (lr_pointer_cast(Int32,vThisIterReady))[lr_cast(Int32,idx_L528)] = _lr_rhs; _lr_rhs)
        end
    end
    
    return nothing
end

# Original gatherByScannedMask source device body; candidate, not certified.
@inline function lr_gatherByScannedMask(ctx,vScannedMask,vGathered,arrayLen,idOffset)::Nothing
    idx_L569=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L569) < lr_cast(Int32,arrayLen))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L569) == lr_cast(Int32,0))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vScannedMask))[lr_cast(Int32,0)]) == lr_cast(Int32,1))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idOffset)); (lr_pointer_cast(Int32,vGathered))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
            end
        else
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vScannedMask))[lr_cast(Int32,idx_L569)]) > lr_cast(Int32,(lr_pointer_cast(Int32,vScannedMask))[lr_cast(Int32,(lr_cast(Int32,idx_L569) - lr_cast(Int32,1)))]))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L569) + lr_cast(Int32,idOffset)))); (lr_pointer_cast(Int32,vGathered))[lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,vScannedMask))[lr_cast(Int32,idx_L569)]) - lr_cast(Int32,1)))] = _lr_rhs; _lr_rhs)
            end
        end
    end
    
    return nothing
end

# Original markIsCoverOutput source device body; candidate, not certified.
@inline function lr_markIsCoverOutput(ctx,vCoverTableLens,vMarks,arrayLen)::Nothing
    idx_L620=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L620) < lr_cast(Int32,arrayLen))))
        (_lr_rhs=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLens))[lr_cast(Int32,idx_L620)]) != lr_cast(Int32,(-lr_cast(Int32,1)))))) ? lr_cast(Int32,1) : lr_cast(Int32,0)))); (lr_pointer_cast(Int32,vMarks))[lr_cast(Int32,idx_L620)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original getCoverToNodeIdMapping source device body; candidate, not certified.
@inline function lr_getCoverToNodeIdMapping(ctx,vNodeCoverIdMapping,vCoverTableLens,vCoverNodeIdMapping,arrayLen)::Nothing
    idx_L628=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L628) < lr_cast(Int32,arrayLen))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLens))[lr_cast(Int32,idx_L628)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idx_L628)); (lr_pointer_cast(Int32,vCoverNodeIdMapping))[lr_cast(Int32,(lr_pointer_cast(Int32,vNodeCoverIdMapping))[lr_cast(Int32,idx_L628)])] = _lr_rhs; _lr_rhs)
        end
    end
    
    return nothing
end

# Original gatherWithFilter source device body; candidate, not certified.
@inline function lr_gatherWithFilter(ctx,isEnd,vCoverTable,vCoverTableLinks,vNodes,vCoverRanges,vCanonTable,vNewGlobalList,vNodesFilter,nPIs,arrayLen)::Nothing
    idx_L642=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L642) < lr_cast(Int32,arrayLen))))
        dstStartIdx_L644=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L642) == lr_cast(Int32,0)))) ? lr_cast(Int32,0) : lr_cast(Int32,(lr_pointer_cast(Int32,vCoverRanges))[lr_cast(Int32,(lr_cast(Int32,idx_L642) - lr_cast(Int32,1)))]))))
        dstEndIdx_L646=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverRanges))[lr_cast(Int32,idx_L642)]))
        fatherLit_L648=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vNodes))[lr_cast(Int32,idx_L642)]))
        fatherId_L649=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,fatherLit_L648)))
        local currRowIdx_L655
        local columnPtr_L655
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,fatherId_L649)); currRowIdx_L655 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L655 = _lr_rhs; _lr_rhs)
        local laterThread_L659
        local currLit_L659
        local currId_L659
        i_L660=lr_cast(Int32,lr_cast(Int32,dstStartIdx_L644))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L660) < lr_cast(Int32,dstEndIdx_L646))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,columnPtr_L655) == lr_cast(Int32,lr_cast(Int32,4)))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L655 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,currRowIdx_L655)])); currRowIdx_L655 = _lr_rhs; _lr_rhs)
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L655) * lr_cast(Int32,lr_cast(Int32,4)))) + lr_cast(Int32,columnPtr_L655)))])); currLit_L659 = _lr_rhs; _lr_rhs)
            (_lr_old=columnPtr_L655; columnPtr_L655=lr_cast(Int32,(columnPtr_L655+Int32(1))); _lr_old)
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,currLit_L659))); currId_L659 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currLit_L659)); (lr_pointer_cast(Int32,vNewGlobalList))[lr_cast(Int32,i_L660)] = _lr_rhs; _lr_rhs)
            if lr_truth((!lr_truth(lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,currId_L659),lr_cast(Int32,nPIs))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vNodesFilter))[lr_cast(Int32,i_L660)] = _lr_rhs; _lr_rhs)
                (_lr_old=i_L660; i_L660=lr_cast(Int32,(i_L660+Int32(1))); _lr_old); continue
            end
            (_lr_rhs=lr_cast(Int32,lr_atomicCAS((lr_pointer_cast(Int32,vCanonTable) + lr_cast(Int32,currId_L659)),lr_cast(Int32,0),lr_cast(Int32,1))); laterThread_L659 = _lr_rhs; _lr_rhs)
            if lr_truth((!lr_truth(lr_cast(Bool,lr_cast(Int32,laterThread_L659)))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vNodesFilter))[lr_cast(Int32,i_L660)] = _lr_rhs; _lr_rhs)
            else
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); (lr_pointer_cast(Int32,vNodesFilter))[lr_cast(Int32,i_L660)] = _lr_rhs; _lr_rhs)
            end
            (_lr_old=i_L660; i_L660=lr_cast(Int32,(i_L660+Int32(1))); _lr_old)
        end
    end
    
    return nothing
end

# Original localCoverTravToTable source device body; candidate, not certified.
@inline function lr_localCoverTravToTable(ctx,nodeLit,pFanin0,pFanin1,pNumFanouts,vCoverTable,vCoverTableLinks,vCoverTableLens,nCoverTableNext,superLen,nPIs)::Int32
    _lr_mem_0=lr_alloc(LRArray{Int32,128})
    stack_L705=lr_pointer_cast(Int32,_lr_mem_0)
    local nodeId_L700
    local outputNodeId_L700
    local i_L701
    local currLit_L701
    local checkLit_L701
    local samePolarFlag_L701
    local currRowIdx_L702
    local lastRowIdx_L702
    local columnPtr_L702
    local checkCurrRowIdx_L703
    local checkColumnPtr_L703

    stackTop_L706=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeLit)); (lr_pointer_cast(Int32,stack_L705))[lr_cast(Int32,(stackTop_L706=lr_cast(Int32,(stackTop_L706+Int32(1))); stackTop_L706))] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,nodeLit))); outputNodeId_L700 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,outputNodeId_L700)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,outputNodeId_L700)); currRowIdx_L702 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L702 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,stackTop_L706) != lr_cast(Int32,(-lr_cast(Int32,1))))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,stack_L705))[(_lr_old=stackTop_L706; stackTop_L706=lr_cast(Int32,(stackTop_L706-Int32(1))); _lr_old)])); currLit_L701 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); samePolarFlag_L701 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,outputNodeId_L700)); checkCurrRowIdx_L703 = _lr_rhs; _lr_rhs)
        ((_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L701 = _lr_rhs; _lr_rhs); (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); checkColumnPtr_L703 = _lr_rhs; _lr_rhs))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L701) < lr_cast(Int32,(lr_pointer_cast(Int32,superLen))[0]))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,checkColumnPtr_L703) == lr_cast(Int32,lr_cast(Int32,4)))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); checkColumnPtr_L703 = _lr_rhs; _lr_rhs)
                ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,checkCurrRowIdx_L703)]) > lr_cast(Int32,checkCurrRowIdx_L703))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,checkCurrRowIdx_L703)])); checkCurrRowIdx_L703 = _lr_rhs; _lr_rhs)
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,checkCurrRowIdx_L703) * lr_cast(Int32,lr_cast(Int32,4)))) + lr_cast(Int32,checkColumnPtr_L703)))])); checkLit_L701 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,checkLit_L701) == lr_cast(Int32,currLit_L701))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); samePolarFlag_L701 = _lr_rhs; _lr_rhs)
                break
            else
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,checkLit_L701) == lr_s_AigNodeNot(lr_cast(Int32,currLit_L701)))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vCoverTableLens))[lr_cast(Int32,outputNodeId_L700)] = _lr_rhs; _lr_rhs)
                    begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
                end
            end
            ((_lr_old=i_L701; i_L701=lr_cast(Int32,(i_L701+Int32(1))); _lr_old); (_lr_old=checkColumnPtr_L703; checkColumnPtr_L703=lr_cast(Int32,(checkColumnPtr_L703+Int32(1))); _lr_old))
        end
        if lr_truth(lr_cast(Bool,lr_cast(Int32,samePolarFlag_L701)))
            continue
        end
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,currLit_L701))); nodeId_L700 = _lr_rhs; _lr_rhs)
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,currLit_L701) != lr_cast(Int32,nodeLit)))) && lr_truth(((lr_truth((lr_truth((lr_truth((lr_truth(lr_cast(Bool,lr_s_AigNodeIsComplement(lr_cast(Int32,currLit_L701)))) || lr_truth((!lr_truth(lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,nodeId_L700),lr_cast(Int32,nPIs)))))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,pNumFanouts))[lr_cast(Int32,nodeId_L700)]) > lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,stackTop_L706) + lr_cast(Int32,2))) >= lr_cast(Int32,lr_cast(Int32,128))))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,superLen))[0]) + lr_cast(Int32,(lr_cast(Int32,2) * (lr_cast(Int32,(lr_cast(Int32,stackTop_L706) + lr_cast(Int32,1)))))))) + lr_cast(Int32,1))) >= lr_cast(Int32,lr_cast(Int32,1024))))))))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,columnPtr_L702) == lr_cast(Int32,lr_cast(Int32,4)))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currRowIdx_L702)); lastRowIdx_L702 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_atomicAdd(lr_pointer_cast(Int32,nCoverTableNext),lr_cast(Int32,1))); currRowIdx_L702 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currRowIdx_L702)); (lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,lastRowIdx_L702)] = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,currRowIdx_L702)] = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L702 = _lr_rhs; _lr_rhs)
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currLit_L701)); (lr_pointer_cast(Int32,vCoverTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L702) * lr_cast(Int32,lr_cast(Int32,4)))) + lr_cast(Int32,columnPtr_L702)))] = _lr_rhs; _lr_rhs)
            (_lr_old=columnPtr_L702; columnPtr_L702=lr_cast(Int32,(columnPtr_L702+Int32(1))); _lr_old)
            (_lr_old=((lr_pointer_cast(Int32,superLen))[0]); ((lr_pointer_cast(Int32,superLen))[0])=lr_cast(Int32,(((lr_pointer_cast(Int32,superLen))[0])+Int32(1))); _lr_old)
            continue
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin1))[lr_cast(Int32,nodeId_L700)])); (lr_pointer_cast(Int32,stack_L705))[lr_cast(Int32,(stackTop_L706=lr_cast(Int32,(stackTop_L706+Int32(1))); stackTop_L706))] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,pFanin0))[lr_cast(Int32,nodeId_L700)])); (lr_pointer_cast(Int32,stack_L705))[lr_cast(Int32,(stackTop_L706=lr_cast(Int32,(stackTop_L706+Int32(1))); stackTop_L706))] = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,stackTop_L706) < lr_cast(Int32,lr_cast(Int32,128)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,superLen))[0])); (lr_pointer_cast(Int32,vCoverTableLens))[lr_cast(Int32,outputNodeId_L700)] = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end

# Original coverFindingToTable source device body; candidate, not certified.
@inline function lr_coverFindingToTable(ctx,vNodes,vNodesStatus,vLastAppearLevel,pFanin0,pFanin1,pNumFanouts,vCoverTable,vCoverTableLinks,vCoverTableLens,nCoverTableNext,nPIs,arrayLen,levelCount)::Nothing
    _lr_mem_0=lr_alloc(Int32)
    superLen_L797=_lr_mem_0
    idx_L785=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L785) < lr_cast(Int32,arrayLen))))
        nodeId_L795=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vNodes))[lr_cast(Int32,idx_L785)]))
        thisLit_L796=lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,nodeId_L795) << lr_cast(Int32,1)))))
        superLen_L797[0]=lr_cast(Int32,lr_cast(Int32,0))
        local ret_L798
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,levelCount)); (lr_pointer_cast(Int32,vLastAppearLevel))[lr_cast(Int32,nodeId_L795)] = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,nodeId_L795)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLens))[lr_cast(Int32,nodeId_L795)])); superLen_L797[0] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Bool,(lr_cast(Int32,superLen_L797[0]) != lr_cast(Int32,0)))))); ret_L798 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(Int32,lr_localCoverTravToTable(ctx ,lr_cast(Int32,thisLit_L796),lr_pointer_cast(Int32,pFanin0),lr_pointer_cast(Int32,pFanin1),lr_pointer_cast(Int32,pNumFanouts),lr_pointer_cast(Int32,vCoverTable),lr_pointer_cast(Int32,vCoverTableLinks),lr_pointer_cast(Int32,vCoverTableLens),lr_pointer_cast(Int32,nCoverTableNext),superLen_L797,lr_cast(Int32,nPIs))); ret_L798 = _lr_rhs; _lr_rhs)
        end
        if lr_truth((!lr_truth(lr_cast(Bool,lr_cast(Int32,ret_L798)))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vNodes))[lr_cast(Int32,idx_L785)] = _lr_rhs; _lr_rhs)
        else
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,superLen_L797[0]) > lr_cast(Int32,1))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,superLen_L797[0]) <= lr_cast(Int32,lr_cast(Int32,1024)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            local currRowIdx_L837
            local columnPtr_L837
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeId_L795)); currRowIdx_L837 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L837 = _lr_rhs; _lr_rhs)
            local currLit_L841
            local currId_L841
            i_L842=lr_cast(Int32,lr_cast(Int32,0))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L842) < lr_cast(Int32,superLen_L797[0]))))
                if lr_truth(lr_cast(Bool,(lr_cast(Int32,columnPtr_L837) == lr_cast(Int32,lr_cast(Int32,4)))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L837 = _lr_rhs; _lr_rhs)
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTableLinks))[lr_cast(Int32,currRowIdx_L837)])); currRowIdx_L837 = _lr_rhs; _lr_rhs)
                end
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L837) * lr_cast(Int32,lr_cast(Int32,4)))) + lr_cast(Int32,columnPtr_L837)))])); currLit_L841 = _lr_rhs; _lr_rhs)
                (_lr_old=columnPtr_L837; columnPtr_L837=lr_cast(Int32,(columnPtr_L837+Int32(1))); _lr_old)
                (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,currLit_L841))); currId_L841 = _lr_rhs; _lr_rhs)
                if lr_truth(lr_cast(Bool,lr_s_AigIsNode(lr_cast(Int32,currId_L841),lr_cast(Int32,nPIs))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,vNodesStatus))[lr_cast(Int32,currId_L841)] = _lr_rhs; _lr_rhs)
                end
                (_lr_old=i_L842; i_L842=lr_cast(Int32,(i_L842+Int32(1))); _lr_old)
            end
        end
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end

# Original findLevelNodeRanges source device body; candidate, not certified.
@inline function lr_findLevelNodeRanges(ctx,vLastAppearLevel,vLastAppearLevelRanges,nCovers)::Nothing
    idx_L865=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L865) == lr_cast(Int32,(lr_cast(Int32,nCovers) - lr_cast(Int32,1))))))
        Int32(0)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L865) + lr_cast(Int32,1)))); (lr_pointer_cast(Int32,vLastAppearLevelRanges))[lr_cast(Int32,(lr_pointer_cast(Int32,vLastAppearLevel))[lr_cast(Int32,idx_L865)])] = _lr_rhs; _lr_rhs)
    else
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L865) < lr_cast(Int32,(lr_cast(Int32,nCovers) - lr_cast(Int32,1))))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vLastAppearLevel))[lr_cast(Int32,idx_L865)]) < lr_cast(Int32,(lr_pointer_cast(Int32,vLastAppearLevel))[lr_cast(Int32,(lr_cast(Int32,idx_L865) + lr_cast(Int32,1)))]))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,idx_L865) + lr_cast(Int32,1)))); (lr_pointer_cast(Int32,vLastAppearLevelRanges))[lr_cast(Int32,(lr_pointer_cast(Int32,vLastAppearLevel))[lr_cast(Int32,idx_L865)])] = _lr_rhs; _lr_rhs)
            end
        end
    end
    
    return nothing
end

# Original parseOutputRes source device body; candidate, not certified.
@inline function lr_parseOutputRes(ctx,vReconstructedKeys,htOutKeys,htOutValues,htOutCapacity,vFanin0New,vFanin1New,vNumFanoutsNew,nEntries,nPIs)::Nothing
    _lr_mem_0=lr_alloc(UInt32)
    oldLit0_L885=_lr_mem_0
    _lr_mem_1=lr_alloc(UInt32)
    oldLit1_L885=_lr_mem_1
    idx_L882=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L882) < lr_cast(Int32,nEntries))))
        key_L884=lr_cast(UInt64,lr_cast(UInt64,(lr_pointer_cast(UInt64,vReconstructedKeys))[lr_cast(Int32,idx_L882)]))


        local oldId0_L885
        local oldId1_L885
        local newId0_L885
        local newId1_L885
        lr_s_unbindAndNodeKeys(lr_cast(UInt64,key_L884),oldLit0_L885,oldLit1_L885)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_s_AigNodeID(lr_cast(Int32,lr_cast(UInt32,oldLit0_L885[0]))))); oldId0_L885 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_s_AigNodeID(lr_cast(Int32,lr_cast(UInt32,oldLit1_L885[0]))))); oldId1_L885 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,lr_s_AigIsPIConst(lr_cast(Int32,lr_cast(UInt32,oldId0_L885)),lr_cast(Int32,nPIs))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,oldId0_L885)); newId0_L885 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(UInt32,lr_s_retrieve_single(lr_pointer_cast(UInt32,htOutKeys),lr_pointer_cast(UInt32,htOutValues),lr_cast(UInt32,oldId0_L885),lr_cast(Int32,htOutCapacity))); newId0_L885 = _lr_rhs; _lr_rhs)
        end
        if lr_truth(lr_cast(Bool,lr_s_AigIsPIConst(lr_cast(Int32,lr_cast(UInt32,oldId1_L885)),lr_cast(Int32,nPIs))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,oldId1_L885)); newId1_L885 = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(UInt32,lr_s_retrieve_single(lr_pointer_cast(UInt32,htOutKeys),lr_pointer_cast(UInt32,htOutValues),lr_cast(UInt32,oldId1_L885),lr_cast(Int32,htOutCapacity))); newId1_L885 = _lr_rhs; _lr_rhs)
        end
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,newId0_L885) != lr_cast(UInt32,(typemax(UInt32))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,newId1_L885) != lr_cast(UInt32,(typemax(UInt32))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,lr_cast(UInt32,newId0_L885)),lr_s_AigNodeIsComplement(lr_cast(Int32,lr_cast(UInt32,oldLit0_L885[0]))))); (lr_pointer_cast(Int32,vFanin0New))[lr_cast(Int32,idx_L882)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,lr_cast(UInt32,newId1_L885)),lr_s_AigNodeIsComplement(lr_cast(Int32,lr_cast(UInt32,oldLit1_L885[0]))))); (lr_pointer_cast(Int32,vFanin1New))[lr_cast(Int32,idx_L882)] = _lr_rhs; _lr_rhs)
        lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(UInt32,newId0_L885)),lr_cast(Int32,1))
        lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(UInt32,newId1_L885)),lr_cast(Int32,1))
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original processPO source device body; candidate, not certified.
@inline function lr_processPO(ctx,d_pOuts,vNodeCoverIdMapping,vCoverNodeNewLitMapping,htOutKeys,htOutValues,htOutCapacity,vNewOuts,vNumFanoutsNew,nPOs,nPIs)::Nothing
    idx_L913=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L913) < lr_cast(Int32,nPOs))))
        oldLit_L915=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,d_pOuts))[lr_cast(Int32,idx_L913)]))
        oldId_L916=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,oldLit_L915)))
        if lr_truth(lr_cast(Bool,lr_s_AigIsPIConst(lr_cast(Int32,oldId_L916),lr_cast(Int32,nPIs))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,oldLit_L915)); (lr_pointer_cast(Int32,vNewOuts))[lr_cast(Int32,idx_L913)] = _lr_rhs; _lr_rhs)
            lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(Int32,oldId_L916)),lr_cast(Int32,1))
        else
            newLit_L923=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCoverNodeNewLitMapping))[lr_cast(Int32,(lr_pointer_cast(Int32,vNodeCoverIdMapping))[lr_cast(Int32,oldId_L916)])]))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,newLit_L923) != lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            outId_L925=lr_cast(Int32,lr_cast(Int32,lr_s_retrieve_single(lr_pointer_cast(UInt32,htOutKeys),lr_pointer_cast(UInt32,htOutValues),lr_cast(UInt32,lr_s_AigNodeID(lr_cast(Int32,newLit_L923))),lr_cast(Int32,htOutCapacity))))
            if lr_truth(lr_cast(Bool,(lr_cast(UInt32,lr_cast(Int32,outId_L925)) == lr_cast(UInt32,(typemax(UInt32))))))
                Int32(0)
                (_lr_rhs=lr_cast(Int32,lr_s_AigNodeID(lr_cast(Int32,newLit_L923))); outId_L925 = _lr_rhs; _lr_rhs)
                ((lr_truth(lr_cast(Bool,lr_cast(Bool,lr_s_AigIsPIConst(lr_cast(Int32,outId_L925),lr_cast(Int32,nPIs))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            end
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(UInt32,lr_cast(Int32,outId_L925)) != lr_cast(UInt32,(typemax(UInt32))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(Int32,lr_s_AigNodeLitCond(lr_cast(Int32,outId_L925),lr_cast(Int32,xor(lr_s_AigNodeIsComplement(lr_cast(Int32,oldLit_L915)),lr_s_AigNodeIsComplement(lr_cast(Int32,newLit_L923)))))); (lr_pointer_cast(Int32,vNewOuts))[lr_cast(Int32,idx_L913)] = _lr_rhs; _lr_rhs)
            lr_atomicAdd((lr_pointer_cast(Int32,vNumFanoutsNew) + lr_cast(Int32,outId_L925)),lr_cast(Int32,1))
        end
    end
    
    return nothing
end
