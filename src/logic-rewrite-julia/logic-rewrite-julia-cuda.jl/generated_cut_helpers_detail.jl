# GENERATED from original source; full-pipeline verification PENDING.
# Original CutFindValue source device body; candidate, not certified.
@inline function lr_CutFindValue(ctx,cut,nRef)::Int32
    value_L80=lr_cast(Int32,lr_cast(Int32,0))
    nOnes_L80=lr_cast(Int32,lr_cast(Int32,0))
    i_L81=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L81) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(value_L80 + lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L81)])])))); value_L80 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(nOnes_L80 + lr_cast(Int32,(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L81)])]) == lr_cast(Int32,1)))))))); nOnes_L80 = _lr_rhs; _lr_rhs)
        (_lr_old=i_L81; i_L81=lr_cast(Int32,(i_L81+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)) < lr_cast(Int32,2))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,1001)); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,value_L80) > lr_cast(Int32,1000))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1000)); value_L80 = _lr_rhs; _lr_rhs)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nOnes_L80) > lr_cast(Int32,3))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,5) - lr_cast(Int32,nOnes_L80)))); value_L80 = _lr_rhs; _lr_rhs)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,value_L80)); ; return _lr_result end
    
    return nothing
end

# Original Inputs source device body; candidate, not certified.
@inline function lr_Inputs(ctx,nRef,cuts,n)::Nothing
    idx_L94=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1))))) + lr_cast(UInt32,lr_cast(Int32,1))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L94) <= lr_cast(Int32,n))))
        cut_L96=lr_pointer_cast(LRCut,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cuts) + (lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx_L94)) * lr_cast(Int32,8))) + (lr_cast(Int32,0))))))))
        (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L96)).used = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,idx_L94) & lr_cast(Int32,31))))))); (lr_pointer_cast(LRCut,cut_L96)).sign = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,43690))); (lr_pointer_cast(LRCut,cut_L96)).truthtable = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L96)).nLeaves = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idx_L94)); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L96)).leaves))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_CutFindValue(ctx ,lr_pointer_cast(LRCut,cut_L96),lr_pointer_cast(Int32,nRef)))); (lr_pointer_cast(LRCut,cut_L96)).value = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original CountOnes source device body; candidate, not certified.
@inline function lr_CountOnes(ctx,uWord)::Int32
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,1431655765))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,1)))) & lr_cast(UInt32,lr_cast(Int32,1431655765)))))))); uWord = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,858993459))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,2)))) & lr_cast(UInt32,lr_cast(Int32,858993459)))))))); uWord = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,252645135))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,4)))) & lr_cast(UInt32,lr_cast(Int32,252645135)))))))); uWord = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,16711935))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,8)))) & lr_cast(UInt32,lr_cast(Int32,16711935)))))))); uWord = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,65535))))) + (lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,16)))))))); ; return _lr_result end
    
    return nothing
end

# Original FindCut source device body; candidate, not certified.
@inline function lr_FindCut(ctx,idx,cuts)::Int32
    i_L115=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L115) < lr_cast(Int32,8))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(Bool,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L115)))))]).used)) == lr_cast(Int32,0))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,i_L115)); ; return _lr_result end
        end
        (_lr_old=i_L115; i_L115=lr_cast(Int32,(i_L115+Int32(1))); _lr_old)
    end
    ans_L117=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    i_L118=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L118) < lr_cast(Int32,8))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L118)))))]).nLeaves)) > lr_cast(Int32,2))))
            if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L118)))))]).value)) < lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).value)))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L118)); ans_L117 = _lr_rhs; _lr_rhs)
            end
        end
        (_lr_old=i_L118; i_L118=lr_cast(Int32,(i_L118+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        i_L124=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,8))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L124)))))]).nLeaves)) == lr_cast(Int32,2))))
                if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L124)))))]).value)) < lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).value)))))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L124)); ans_L117 = _lr_rhs; _lr_rhs)
                end
            end
            (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
        end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        i_L131=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L131) < lr_cast(Int32,8))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L131)))))]).nLeaves)) < lr_cast(Int32,2))))
                if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L131)))))]).value)) < lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).value)))))))
                    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L131)); ans_L117 = _lr_rhs; _lr_rhs)
                end
            end
            (_lr_old=i_L131; i_L131=lr_cast(Int32,(i_L131+Int32(1))); _lr_old)
        end
    end
    (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).used = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,ans_L117)); ; return _lr_result end
    
    return nothing
end

# Original MergeCutOrdered source device body; candidate, not certified.
@inline function lr_MergeCutOrdered(ctx,a,b,cut)::Int32
    _lr_mem_0=lr_alloc(LRCut)
    lr_copy!(_lr_mem_0,a,LRCut)
    a=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    lr_copy!(_lr_mem_1,b,LRCut)
    b=_lr_mem_1[0]
    c_L143=lr_cast(Int32,lr_cast(Int32,0))
    i_L143=lr_cast(Int32,lr_cast(Int32,0))
    k_L143=lr_cast(Int32,lr_cast(Int32,0))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); c_L143 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,c_L143) < lr_cast(Int32,4))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L143) == lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L143) == lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,c_L143))); (lr_pointer_cast(LRCut,cut)).nLeaves = _lr_rhs; _lr_rhs)
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[(_lr_old=i_L143; i_L143=lr_cast(Int32,(i_L143+Int32(1))); _lr_old)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = _lr_rhs; _lr_rhs)
            (_lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old); continue
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L143) == lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L143) == lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,c_L143))); (lr_pointer_cast(LRCut,cut)).nLeaves = _lr_rhs; _lr_rhs)
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[(_lr_old=k_L143; k_L143=lr_cast(Int32,(k_L143+Int32(1))); _lr_old)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = _lr_rhs; _lr_rhs)
            (_lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old); continue
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[lr_cast(Int32,i_L143)]) < lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[lr_cast(Int32,k_L143)]))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[(_lr_old=i_L143; i_L143=lr_cast(Int32,(i_L143+Int32(1))); _lr_old)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = _lr_rhs; _lr_rhs)
            (_lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old); continue
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[lr_cast(Int32,i_L143)]) > lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[lr_cast(Int32,k_L143)]))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[(_lr_old=k_L143; k_L143=lr_cast(Int32,(k_L143+Int32(1))); _lr_old)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = _lr_rhs; _lr_rhs)
            (_lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old); continue
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[(_lr_old=i_L143; i_L143=lr_cast(Int32,(i_L143+Int32(1))); _lr_old)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = _lr_rhs; _lr_rhs)
        (_lr_old=k_L143; k_L143=lr_cast(Int32,(k_L143+Int32(1))); _lr_old)
        (_lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old)
    end
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L143) < lr_cast(Int32,lr_cast(UInt32,(a).nLeaves))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L143) < lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    end
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,c_L143))); (lr_pointer_cast(LRCut,cut)).nLeaves = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original MergeCut source device body; candidate, not certified.
@inline function lr_MergeCut(ctx,a,b,cut)::Int32
    _lr_mem_0=lr_alloc(LRCut)
    lr_copy!(_lr_mem_0,a,LRCut)
    a=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    lr_copy!(_lr_mem_1,b,LRCut)
    b=_lr_mem_1[0]
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)) >= lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))
        if lr_truth(lr_cast(Bool,(lr_MergeCutOrdered(ctx ,a,b,lr_pointer_cast(LRCut,cut)) == lr_cast(Int32,0))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
        end
    else
        if lr_truth(lr_cast(Bool,(lr_MergeCutOrdered(ctx ,b,a,lr_pointer_cast(LRCut,cut)) == lr_cast(Int32,0))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
        end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(a).sign) | lr_cast(Int32,(b).sign)))); (lr_pointer_cast(LRCut,cut)).sign = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut)).used = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original CutDominance source device body; candidate, not certified.
@inline function lr_CutDominance(ctx,a,b)::Int32
    _lr_mem_0=lr_alloc(LRCut)
    lr_copy!(_lr_mem_0,a,LRCut)
    a=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    lr_copy!(_lr_mem_1,b,LRCut)
    b=_lr_mem_1[0]
    i_L198=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L198) < lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)))))
        ok_L199=lr_cast(Int32,lr_cast(Int32,0))
        j_L200=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L200) < lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[lr_cast(Int32,j_L200)]) == lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[lr_cast(Int32,i_L198)]))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); ok_L199 = _lr_rhs; _lr_rhs)
            end
            (_lr_old=j_L200; j_L200=lr_cast(Int32,(j_L200+Int32(1))); _lr_old)
        end
        if lr_truth((!lr_truth(lr_cast(Bool,lr_cast(Int32,ok_L199)))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
        end
        (_lr_old=i_L198; i_L198=lr_cast(Int32,(i_L198+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original CutFilter source device body; candidate, not certified.
@inline function lr_CutFilter(ctx,cut,id)::Int32
    i_L209=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L209) < lr_cast(Int32,8))))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L209) != lr_cast(Int32,id)))) && lr_truth(lr_cast(Bool,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).used))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).nLeaves)) > lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).nLeaves)))))
                if lr_truth(lr_cast(Bool,((lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).sign) & lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).sign)))) != lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).sign))))
                    (_lr_old=i_L209; i_L209=lr_cast(Int32,(i_L209+Int32(1))); _lr_old); continue
                end
                if lr_truth(lr_cast(Bool,lr_CutDominance(ctx ,(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)],(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)])))
                    (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).used = _lr_rhs; _lr_rhs)
                end
            else
                if lr_truth(lr_cast(Bool,((lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).sign) & lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).sign)))) != lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).sign))))
                    (_lr_old=i_L209; i_L209=lr_cast(Int32,(i_L209+Int32(1))); _lr_old); continue
                end
                if lr_truth(lr_cast(Bool,lr_CutDominance(ctx ,(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)],(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)])))
                    (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).used = _lr_rhs; _lr_rhs)
                    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
                end
            end
        end
        (_lr_old=i_L209; i_L209=lr_cast(Int32,(i_L209+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    
    return nothing
end

# Original CutTruthPhase source device body; candidate, not certified.
@inline function lr_CutTruthPhase(ctx,x,cut)::Int32
    _lr_mem_0=lr_alloc(LRCut)
    lr_copy!(_lr_mem_0,x,LRCut)
    x=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    lr_copy!(_lr_mem_1,cut,LRCut)
    cut=_lr_mem_1[0]
    phase_L226=lr_cast(Int32,lr_cast(Int32,0))
    i_L227=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L227) < lr_cast(Int32,lr_cast(UInt32,(cut).nLeaves)))))
        j_L228=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L228) < lr_cast(Int32,lr_cast(UInt32,(x).nLeaves)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,(x).leaves))[lr_cast(Int32,j_L228)]) == lr_cast(Int32,(lr_pointer_cast(Int32,(cut).leaves))[lr_cast(Int32,i_L227)]))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(phase_L226 | lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L227)))))); phase_L226 = _lr_rhs; _lr_rhs)
            end
            (_lr_old=j_L228; j_L228=lr_cast(Int32,(j_L228+Int32(1))); _lr_old)
        end
        (_lr_old=i_L227; i_L227=lr_cast(Int32,(i_L227+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,phase_L226)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original CutTruthSwapAdjacentVars source device body; candidate, not certified.
@inline function lr_CutTruthSwapAdjacentVars(ctx,uTruth,iVar)::Int32
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,iVar) == lr_cast(Int32,0))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,lr_cast(Int32,uTruth)) & lr_cast(UInt32,2576980377)))) | lr_cast(UInt32,(lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,uTruth) & lr_cast(Int32,572662306)))) << lr_cast(Int32,1))))))) | lr_cast(UInt32,(lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,uTruth) & lr_cast(Int32,1145324612)))) >> lr_cast(Int32,1))))))))); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,iVar) == lr_cast(Int32,1))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,lr_cast(Int32,uTruth)) & lr_cast(UInt32,3284386755)))) | lr_cast(UInt32,(lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,uTruth) & lr_cast(Int32,202116108)))) << lr_cast(Int32,2))))))) | lr_cast(UInt32,(lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,uTruth) & lr_cast(Int32,808464432)))) >> lr_cast(Int32,2))))))))); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,iVar) == lr_cast(Int32,2))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,lr_cast(Int32,uTruth)) & lr_cast(UInt32,4027576335)))) | lr_cast(UInt32,(lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,uTruth) & lr_cast(Int32,15728880)))) << lr_cast(Int32,4))))))) | lr_cast(UInt32,(lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,uTruth) & lr_cast(Int32,251662080)))) >> lr_cast(Int32,4))))))))); ; return _lr_result end
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    
    return nothing
end

# Original CutTruthStretch source device body; candidate, not certified.
@inline function lr_CutTruthStretch(ctx,truthtable,nVar,phase)::Int32
    Var_L244=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,nVar) - lr_cast(Int32,1))))
    i_L245=lr_cast(Int32,lr_cast(Int32,3))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L245) >= lr_cast(Int32,0))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,phase) >> lr_cast(Int32,i_L245))) & lr_cast(Int32,1)))))
            k_L246=lr_cast(Int32,lr_cast(Int32,Var_L244))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L246) < lr_cast(Int32,i_L245))))
                (_lr_rhs=lr_cast(Int32,lr_CutTruthSwapAdjacentVars(ctx ,lr_cast(Int32,truthtable),lr_cast(Int32,k_L246))); truthtable = _lr_rhs; _lr_rhs)
                (_lr_old=k_L246; k_L246=lr_cast(Int32,(k_L246+Int32(1))); _lr_old)
            end
            (_lr_old=Var_L244; Var_L244=lr_cast(Int32,(Var_L244-Int32(1))); _lr_old)
        end
        (_lr_old=i_L245; i_L245=lr_cast(Int32,(i_L245-Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,truthtable)); ; return _lr_result end
    
    return nothing
end

# Original CutTruthtable source device body; candidate, not certified.
@inline function lr_CutTruthtable(ctx,cut,a,b,aComplement,bComplement)::Int32
    _lr_mem_0=lr_alloc(LRCut)
    lr_copy!(_lr_mem_0,cut,LRCut)
    cut=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    lr_copy!(_lr_mem_1,a,LRCut)
    a=_lr_mem_1[0]
    _lr_mem_2=lr_alloc(LRCut)
    lr_copy!(_lr_mem_2,b,LRCut)
    b=_lr_mem_2[0]
    tt0_L254=lr_cast(Int32,(lr_truth(lr_cast(Bool,lr_cast(Int32,aComplement))) ? lr_cast(Int32,(~lr_cast(Int32,lr_cast(UInt32,(a).truthtable)))) : lr_cast(Int32,lr_cast(UInt32,(a).truthtable))))
    tt1_L255=lr_cast(Int32,(lr_truth(lr_cast(Bool,lr_cast(Int32,bComplement))) ? lr_cast(Int32,(~lr_cast(Int32,lr_cast(UInt32,(b).truthtable)))) : lr_cast(Int32,lr_cast(UInt32,(b).truthtable))))
    (_lr_rhs=lr_cast(Int32,lr_CutTruthStretch(ctx ,lr_cast(Int32,tt0_L254),lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)),lr_CutTruthPhase(ctx ,a,cut))); tt0_L254 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_CutTruthStretch(ctx ,lr_cast(Int32,tt1_L255),lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)),lr_CutTruthPhase(ctx ,b,cut))); tt1_L255 = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,tt0_L254) & lr_cast(Int32,tt1_L255)))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base)))
    return nothing
end

# Original CutTruthShrink source device body; candidate, not certified.
@inline function lr_CutTruthShrink(ctx,uTruth,nVars,Phase)::Int32
    local i_L262
    local k_L262
    Var_L262=lr_cast(Int32,lr_cast(Int32,0))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L262 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L262) < lr_cast(Int32,4))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,Phase) & (lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L262))))))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,i_L262) - lr_cast(Int32,1)))); k_L262 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L262) >= lr_cast(Int32,Var_L262))))
                (_lr_rhs=lr_cast(Int32,lr_CutTruthSwapAdjacentVars(ctx ,lr_cast(Int32,uTruth),lr_cast(Int32,k_L262))); uTruth = _lr_rhs; _lr_rhs)
                (_lr_old=k_L262; k_L262=lr_cast(Int32,(k_L262-Int32(1))); _lr_old)
            end
            (_lr_old=Var_L262; Var_L262=lr_cast(Int32,(Var_L262+Int32(1))); _lr_old)
        end
        (_lr_old=i_L262; i_L262=lr_cast(Int32,(i_L262+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,uTruth)); ; return _lr_result end
    
    return nothing
end

# Original MinimizeCutSupport source device body; candidate, not certified.
@inline function lr_MinimizeCutSupport(ctx,cut)::Int32
    _lr_mem_0=lr_alloc(LRArray{LRArray{Int32,2},4})
    masks_L274=lr_pointer_cast(LRArray{Int32,2},_lr_mem_0)
    LRPtr{Int32}(getfield(masks_L274,:base))[0]=lr_cast(Int32,21845)
    LRPtr{Int32}(getfield(masks_L274,:base))[1]=lr_cast(Int32,43690)
    LRPtr{Int32}(getfield(masks_L274,:base))[2]=lr_cast(Int32,13107)
    LRPtr{Int32}(getfield(masks_L274,:base))[3]=lr_cast(Int32,52428)
    LRPtr{Int32}(getfield(masks_L274,:base))[4]=lr_cast(Int32,3855)
    LRPtr{Int32}(getfield(masks_L274,:base))[5]=lr_cast(Int32,61680)
    LRPtr{Int32}(getfield(masks_L274,:base))[6]=lr_cast(Int32,255)
    LRPtr{Int32}(getfield(masks_L274,:base))[7]=lr_cast(Int32,65280)
    phase_L280=lr_cast(Int32,lr_cast(Int32,0))
    truth_L280=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).truthtable)) & lr_cast(Int32,65535))))
    nLeaves_L280=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))
    i_L281=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L281) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        if lr_truth(lr_cast(Bool,((lr_cast(Int32,(lr_cast(Int32,truth_L280) & lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,2},masks_L274))[lr_cast(Int32,i_L281)]))[lr_cast(Int32,0)])))) == (lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,truth_L280) & lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,2},masks_L274))[lr_cast(Int32,i_L281)]))[lr_cast(Int32,1)])))) >> (lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L281))))))))))
            (_lr_old=nLeaves_L280; nLeaves_L280=lr_cast(Int32,(nLeaves_L280-Int32(1))); _lr_old)
        else
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(phase_L280 | lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L281)))))); phase_L280 = _lr_rhs; _lr_rhs)
        end
        (_lr_old=i_L281; i_L281=lr_cast(Int32,(i_L281+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLeaves_L280) == lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_CutTruthShrink(ctx ,lr_cast(Int32,truth_L280),lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)),lr_cast(Int32,phase_L280))); truth_L280 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(lr_cast(Int32,truth_L280) & lr_cast(Int32,65535))))); (lr_pointer_cast(LRCut,cut)).truthtable = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRCut,cut)).sign = _lr_rhs; _lr_rhs)
    i_L290=lr_cast(Int32,lr_cast(Int32,0))
    k_L290=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L290) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,phase_L280) >> lr_cast(Int32,i_L290))) & lr_cast(Int32,1)))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L290)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[(_lr_old=k_L290; k_L290=lr_cast(Int32,(k_L290+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(LRCut,cut)).sign | lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,31) & lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L290)]))))))))); (lr_pointer_cast(LRCut,cut)).sign = _lr_rhs; _lr_rhs)
        end
        (_lr_old=i_L290; i_L290=lr_cast(Int32,(i_L290+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,nLeaves_L280))); (lr_pointer_cast(LRCut,cut)).nLeaves = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end

# Original CutEnumerate source device body; candidate, not certified.
@inline function lr_CutEnumerate(ctx,fanin0,fanin1,isC0,isC1,nRef,cuts,delta,n)::Nothing
    _lr_mem_0=lr_alloc(LRCut)
    a_L312=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    b_L312=_lr_mem_1[0]
    idx_L299=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L299) < lr_cast(Int32,n))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(idx_L299 + lr_cast(Int32,(lr_cast(Int32,delta) + lr_cast(Int32,1)))))); idx_L299 = _lr_rhs; _lr_rhs)
        cut_L302=lr_pointer_cast(LRCut,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cuts) + (lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx_L299)) * lr_cast(Int32,8))) + (lr_cast(Int32,0))))))))
        (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L302)).used = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,idx_L299) & lr_cast(Int32,31))))))); (lr_pointer_cast(LRCut,cut_L302)).sign = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,43690))); (lr_pointer_cast(LRCut,cut_L302)).truthtable = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L302)).nLeaves = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idx_L299)); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L302)).leaves))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_CutFindValue(ctx ,lr_pointer_cast(LRCut,cut_L302),lr_pointer_cast(Int32,nRef)))); (lr_pointer_cast(LRCut,cut_L302)).value = _lr_rhs; _lr_rhs)
        in0_L309=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,idx_L299)]))
        in1_L309=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,idx_L299)]))
        i_L310=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L310) < lr_cast(Int32,8))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(Bool,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(in0_L309)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L310)))))]).used)) == lr_cast(Int32,1))))
                j_L311=lr_cast(Int32,lr_cast(Int32,0))
                while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L311) < lr_cast(Int32,8))))
                    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(Bool,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(in1_L309)) * lr_cast(Int32,8))) + lr_cast(Int32,(j_L311)))))]).used)) == lr_cast(Int32,1))))
                        lr_copy!(a_L312,(lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(in0_L309)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L310)))))],LRCut)
                        lr_copy!(b_L312,(lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(in1_L309)) * lr_cast(Int32,8))) + lr_cast(Int32,(j_L311)))))],LRCut)
                        if lr_truth(lr_cast(Bool,(lr_CountOnes(ctx ,lr_cast(UInt32,lr_cast(Int32,(lr_cast(Int32,(a_L312).sign) | lr_cast(Int32,(b_L312).sign))))) > lr_cast(Int32,4))))
                            (_lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old); continue
                        end
                        cutId_L314=lr_cast(Int32,lr_FindCut(ctx ,lr_cast(Int32,idx_L299),lr_pointer_cast(LRCut,cuts)))
                        if lr_truth((!lr_truth(lr_cast(Bool,lr_MergeCut(ctx ,a_L312,b_L312,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cut_L302) + lr_cast(Int32,cutId_L314))))))))
                            (_lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old); continue
                        end
                        if lr_truth(lr_cast(Bool,lr_CutFilter(ctx ,lr_pointer_cast(LRCut,cut_L302),lr_cast(Int32,cutId_L314))))
                            (_lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old); continue
                        end
                        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(lr_cast(Int32,65535) & lr_CutTruthtable(ctx ,(lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)],a_L312,b_L312,lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,idx_L299)]),lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,idx_L299)])))))); ((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).truthtable = _lr_rhs; _lr_rhs)
                        if lr_truth(lr_cast(Bool,lr_MinimizeCutSupport(ctx ,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cut_L302) + lr_cast(Int32,cutId_L314))))))
                            lr_CutFilter(ctx ,lr_pointer_cast(LRCut,cut_L302),lr_cast(Int32,cutId_L314))
                        end
                        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_CutFindValue(ctx ,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cut_L302) + lr_cast(Int32,cutId_L314))),lr_pointer_cast(Int32,nRef)))); ((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).value = _lr_rhs; _lr_rhs)
                        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).nLeaves)) < lr_cast(Int32,2))))
                            begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
                        end
                    end
                    (_lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old)
                end
            end
            (_lr_old=i_L310; i_L310=lr_cast(Int32,(i_L310+Int32(1))); _lr_old)
        end
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original Decrease source device body; candidate, not certified.
@inline function lr_Decrease(ctx,id,tableSize,tableId,tableNum,nRef,cut)::Int32
    i_L327=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L327) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,id) == lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L327)]))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
        end
        (_lr_old=i_L327; i_L327=lr_cast(Int32,(i_L327+Int32(1))); _lr_old)
    end
    i_L329=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L329) < lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableId))[lr_cast(Int32,i_L329)]) == lr_cast(Int32,id))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L329)]=lr_cast(Int32,((lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L329)]-Int32(1))); (lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L329)]))); ; return _lr_result end
        end
        (_lr_old=i_L329; i_L329=lr_cast(Int32,(i_L329+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]) == lr_cast(Int32,8))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,id)); (lr_pointer_cast(Int32,tableId))[lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)])] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,id)]) - lr_cast(Int32,1)))); (lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)])] = _lr_rhs; _lr_rhs)
    (_lr_old=(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]; (lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]=lr_cast(Int32,((lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]+Int32(1))); _lr_old)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,id)]) - lr_cast(Int32,1)))); ; return _lr_result end
    
    return nothing
end

# Original IsDeleted source device body; candidate, not certified.
@inline function lr_IsDeleted(ctx,id,tableSize,tableId,tableNum)::Int32
    i_L343=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L343) < lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableId))[lr_cast(Int32,i_L343)]) == lr_cast(Int32,id))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L343)]) == lr_cast(Int32,0))))); ; return _lr_result end
        end
        (_lr_old=i_L343; i_L343=lr_cast(Int32,(i_L343+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    
    return nothing
end

# Original CalcMFFC source device body; candidate, not certified.
@inline function lr_CalcMFFC(ctx,cur,cut,fanin0,fanin1,tableSize,tableId,tableNum,nRef,root)::Int32
    ans_L350=lr_cast(Int32,lr_cast(Int32,1))
    if lr_truth(lr_cast(Bool,(lr_Decrease(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,cur)]),lr_pointer_cast(Int32,tableSize),lr_pointer_cast(Int32,tableId),lr_pointer_cast(Int32,tableNum),lr_pointer_cast(Int32,nRef),lr_pointer_cast(LRCut,cut)) == lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(ans_L350 + lr_CalcMFFC(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,cur)]),lr_pointer_cast(LRCut,cut),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,tableSize),lr_pointer_cast(Int32,tableId),lr_pointer_cast(Int32,tableNum),lr_pointer_cast(Int32,nRef),lr_cast(Int32,root))))); ans_L350 = _lr_rhs; _lr_rhs)
    end
    if lr_truth(lr_cast(Bool,(lr_Decrease(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,cur)]),lr_pointer_cast(Int32,tableSize),lr_pointer_cast(Int32,tableId),lr_pointer_cast(Int32,tableNum),lr_pointer_cast(Int32,nRef),lr_pointer_cast(LRCut,cut)) == lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(ans_L350 + lr_CalcMFFC(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,cur)]),lr_pointer_cast(LRCut,cut),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,tableSize),lr_pointer_cast(Int32,tableId),lr_pointer_cast(Int32,tableNum),lr_pointer_cast(Int32,nRef),lr_cast(Int32,root))))); ans_L350 = _lr_rhs; _lr_rhs)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,ans_L350)); ; return _lr_result end
    
    return nothing
end

# Original TableInsert source device body; candidate, not certified.
@inline function lr_TableInsert(ctx,in0,in1,C0,C1,hashTable,idx,offset)::Nothing
    index_L359=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,idx) - lr_cast(Int32,offset))) - lr_cast(Int32,1))))
    key_L360=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,0)))
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L360 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,in0) * lr_cast(Int32,7937))))))); key_L360 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L360 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,in1) * lr_cast(Int32,2971))))))); key_L360 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L360 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,C0) * lr_cast(Int32,911))))))); key_L360 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L360 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,C1) * lr_cast(Int32,353))))))); key_L360 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,rem(key_L360,lr_cast(UInt64,lr_cast(Int32,ctx.P[0]))))); key_L360 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,idx)); ((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(Int32,(lr_cast(Int32,ctx.P[0]) + lr_cast(Int32,index_L359)))]).val = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,1))
        res_L368=lr_cast(Int32,lr_atomicCAS(lr_field_address((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(UInt64,key_L360)],Val(:next)),lr_cast(Int32,(-lr_cast(Int32,1))),lr_cast(Int32,(lr_cast(Int32,ctx.P[0]) + lr_cast(Int32,index_L359)))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,res_L368) == lr_cast(Int32,(-lr_cast(Int32,1))))))
            break
        else
            (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,res_L368))); key_L360 = _lr_rhs; _lr_rhs)
        end
    end
    
    return nothing
end

# Original TableLookup source device body; candidate, not certified.
@inline function lr_TableLookup(ctx,in0,in1,C0,C1,hashTable,fanin0,fanin1,isC0,isC1)::Int32
    key_L377=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,0)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,in0) > lr_cast(Int32,in1))))
        temp_L379=lr_cast(Int32,lr_cast(Int32,in0))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,in1)); in0 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L379)); in1 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,C0)); temp_L379 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,C1)); C0 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L379)); C1 = _lr_rhs; _lr_rhs)
    end
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L377 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,in0) * lr_cast(Int32,7937))))))); key_L377 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L377 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,in1) * lr_cast(Int32,2971))))))); key_L377 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L377 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,C0) * lr_cast(Int32,911))))))); key_L377 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L377 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,C1) * lr_cast(Int32,353))))))); key_L377 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,rem(key_L377,lr_cast(UInt64,lr_cast(Int32,ctx.P[0]))))); key_L377 = _lr_rhs; _lr_rhs)
    cur_L391=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(UInt64,key_L377)]).next))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,cur_L391) != lr_cast(Int32,(-lr_cast(Int32,1))))))
        val_L392=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(Int32,cur_L391)]).val))
        if lr_truth((lr_truth((lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,val_L392)]) == lr_cast(Int32,in0)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,val_L392)]) == lr_cast(Int32,in1)))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,val_L392)]) == lr_cast(Int32,C0)))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,val_L392)]) == lr_cast(Int32,C1))))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,val_L392)); ; return _lr_result end
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(Int32,cur_L391)]).next)); cur_L391 = _lr_rhs; _lr_rhs)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); ; return _lr_result end
    
    return nothing
end

# Original Eval source device body; candidate, not certified.
@inline function lr_Eval(ctx,cur,match,Class,lib,curId)::Int32
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match))[lr_cast(Int32,cur)]) > lr_cast(Int32,(-lr_cast(Int32,1))))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match))[lr_cast(Int32,cur)]) == lr_cast(Int32,curId))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,curId)); (lr_pointer_cast(Int32,match))[lr_cast(Int32,cur)] = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,1) + lr_Eval(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin0))[lr_cast(Int32,Class)]))[lr_cast(Int32,(lr_cast(Int32,cur) - lr_cast(Int32,4)))]),lr_pointer_cast(Int32,match),lr_cast(Int32,Class),lr_pointer_cast(LRLibrary,lib),lr_cast(Int32,curId)))) + lr_Eval(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin1))[lr_cast(Int32,Class)]))[lr_cast(Int32,(lr_cast(Int32,cur) - lr_cast(Int32,4)))]),lr_pointer_cast(Int32,match),lr_cast(Int32,Class),lr_pointer_cast(LRLibrary,lib),lr_cast(Int32,curId))))); ; return _lr_result end
    
    return nothing
end

# Original EvaluateNode source device body; candidate, not certified.
@inline function lr_EvaluateNode(ctx,sz,bestout,fanin0,fanin1,isC0,isC1,nodeLevels,cuts,selectedCuts,nRef,lib,hashTable,fUseZeros)::Nothing
    _lr_mem_0=lr_alloc(LRArray{Int32,54})
    match_L412=lr_pointer_cast(Int32,_lr_mem_0)
    _lr_mem_1=lr_alloc(Int32)
    tableSize_L412=_lr_mem_1
    _lr_mem_2=lr_alloc(LRArray{Int32,8})
    tableId_L412=lr_pointer_cast(Int32,_lr_mem_2)
    _lr_mem_3=lr_alloc(LRArray{Int32,8})
    tableNum_L412=lr_pointer_cast(Int32,_lr_mem_3)
    _lr_mem_4=lr_alloc(LRArray{Int32,54})
    matchLevel_L413=lr_pointer_cast(Int32,_lr_mem_4)
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1))))) >= lr_cast(UInt32,lr_cast(Int32,sz)))))
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); return _lr_result end
    end
    id_L411=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,lr_cast(Int32,1)) + lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    reduction_L411=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    bestLevel_L411=lr_cast(Int32,lr_cast(Int32,99999999))
    bestCut_L411=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    local bestOut_L411





    i_L414=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L414) < lr_cast(Int32,8))))
        cut_L415=lr_pointer_cast(LRCut,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cuts) + (lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(id_L411)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L414))))))))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(Bool,(lr_pointer_cast(LRCut,cut_L415)).used)) == lr_cast(Int32,0)))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut_L415)).nLeaves)) < lr_cast(Int32,3))))))
            (_lr_old=i_L414; i_L414=lr_cast(Int32,(i_L414+Int32(1))); _lr_old); continue
        end
        nleaves_L417=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut_L415)).nLeaves)))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,nleaves_L417) == lr_cast(Int32,3))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L415)).leaves))[(_lr_old=(lr_pointer_cast(LRCut,cut_L415)).nLeaves; (lr_pointer_cast(LRCut,cut_L415)).nLeaves=lr_cast(UInt32,((lr_pointer_cast(LRCut,cut_L415)).nLeaves+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); tableSize_L412[0] = _lr_rhs; _lr_rhs)
        saved_L421=lr_cast(Int32,lr_CalcMFFC(ctx ,lr_cast(Int32,id_L411),lr_pointer_cast(LRCut,cut_L415),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),tableSize_L412,lr_pointer_cast(Int32,tableId_L412),lr_pointer_cast(Int32,tableNum_L412),lr_pointer_cast(Int32,nRef),lr_cast(Int32,id_L411)))
        uPhase_L422=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).pPhases))[lr_cast(UInt32,(lr_pointer_cast(LRCut,cut_L415)).truthtable)]))
        if id_L411==10 && i_L414==3;CUDA.@cuprintf("TABSIZE %d\n",tableSize_L412[0]);for z in Int32(0):tableSize_L412[0]-Int32(1);CUDA.@cuprintf("TABLE %d %d %d\n",z,tableId_L412[z],tableNum_L412[z]);end;end
        Class_L423=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).pMap))[lr_cast(UInt32,(lr_pointer_cast(LRCut,cut_L415)).truthtable)]))
        pPerm_L424=lr_pointer_cast(Int32,lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,4},(lr_pointer_cast(LRLibrary,lib)).pPerms4))[lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).pPerms))[lr_cast(UInt32,(lr_pointer_cast(LRCut,cut_L415)).truthtable)])]))
        j_L425=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L425) < lr_cast(Int32,54))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); (lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,j_L425)] = _lr_rhs; _lr_rhs))); (lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,j_L425)] = _lr_rhs; _lr_rhs)
            (_lr_old=j_L425; j_L425=lr_cast(Int32,(j_L425+Int32(1))); _lr_old)
        end
        isC_L427=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,0)))
        j_L428=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L428) < lr_cast(Int32,4))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L415)).leaves))[lr_cast(Int32,(lr_pointer_cast(Int32,pPerm_L424))[lr_cast(Int32,j_L428)])])); (lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,j_L428)] = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,nodeLevels))[lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,j_L428)])])); (lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,j_L428)] = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,uPhase_L422) >> lr_cast(Int32,j_L428))) & lr_cast(Int32,1)))))
                (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(isC_L427 | lr_cast(UInt64,lr_cast(Int64,(lr_cast(Int64,1) << lr_cast(Int32,j_L428))))))); isC_L427 = _lr_rhs; _lr_rhs)
            end
            (_lr_old=j_L428; j_L428=lr_cast(Int32,(j_L428+Int32(1))); _lr_old)
        end
        j_L434=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L434) < lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).nNodes))[lr_cast(Int32,Class_L423)]))))
            num_L435=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,j_L434) + lr_cast(Int32,4))))
            in0_L436=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin0))[lr_cast(Int32,Class_L423)]))[lr_cast(Int32,j_L434)]))
            in1_L436=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin1))[lr_cast(Int32,Class_L423)]))[lr_cast(Int32,j_L434)]))
            ((lr_truth(lr_cast(Bool,(lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,in0_L436)]) != lr_cast(Int32,(-lr_cast(Int32,1)))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,in1_L436)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,1) + lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,in0_L436)]) > lr_cast(Int32,(lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,in1_L436)])))) ? (lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,in0_L436)] : (lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,in1_L436)])))))); (lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,num_L435)] = _lr_rhs; _lr_rhs)
            if lr_truth((lr_truth((lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,in0_L436)]) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,in1_L436)]) == lr_cast(Int32,(-lr_cast(Int32,1)))))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,in0_L436)]) == lr_cast(Int32,id_L411)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,in1_L436)]) == lr_cast(Int32,id_L411))))))
                (_lr_old=j_L434; j_L434=lr_cast(Int32,(j_L434+Int32(1))); _lr_old); continue
            end
            nodeId_L441=lr_cast(Int32,lr_TableLookup(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,in0_L436)]),lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,in1_L436)]),lr_cast(Int32,lr_cast(UInt64,((lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,isC_L427) >> lr_cast(Int32,in0_L436))) & lr_cast(UInt64,lr_cast(Int32,1))))) ^ lr_cast(UInt64,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).isC0))[lr_cast(Int32,Class_L423)]))[lr_cast(Int32,j_L434)]))))),lr_cast(Int32,lr_cast(UInt64,((lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,isC_L427) >> lr_cast(Int32,in1_L436))) & lr_cast(UInt64,lr_cast(Int32,1))))) ^ lr_cast(UInt64,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).isC1))[lr_cast(Int32,Class_L423)]))[lr_cast(Int32,j_L434)]))))),lr_pointer_cast(LRTableNode,hashTable),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,isC0),lr_pointer_cast(Int32,isC1)))
            if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,nodeId_L441) != lr_cast(Int32,(-lr_cast(Int32,1)))))) && lr_truth((!lr_truth(lr_cast(Bool,lr_IsDeleted(ctx ,lr_cast(Int32,nodeId_L441),tableSize_L412,lr_pointer_cast(Int32,tableId_L412),lr_pointer_cast(Int32,tableNum_L412))))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeId_L441)); (lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,num_L435)] = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,nodeLevels))[lr_cast(Int32,nodeId_L441)])); (lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,num_L435)] = _lr_rhs; _lr_rhs)
            end
            (_lr_old=j_L434; j_L434=lr_cast(Int32,(j_L434+Int32(1))); _lr_old)
        end
        out_L447=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,out_L447) < lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).nSubgr))[lr_cast(Int32,Class_L423)]))))
            rt_L448=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,20},(lr_pointer_cast(LRLibrary,lib)).pSubgr))[lr_cast(Int32,Class_L423)]))[lr_cast(Int32,out_L447)]))
            if id_L411==10 && i_L414==3 && out_L447==1;for z in Int32(0):Int32(53);CUDA.@cuprintf("MATCH %d %d\n",z,match_L412[z]);end;end
            if id_L411==10 || id_L411==13; CUDA.@cuprintf("TRACE0 %d %d %d %d %d %d %d\n",id_L411,i_L414,out_L447,saved_L421,Class_L423,rt_L448,match_L412[rt_L448]);end
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L412))[lr_cast(Int32,rt_L448)]) == lr_cast(Int32,id_L411))))
                (_lr_old=out_L447; out_L447=lr_cast(Int32,(out_L447+Int32(1))); _lr_old); continue
            end
            nodesAdded_L450=lr_cast(Int32,lr_Eval(ctx ,lr_cast(Int32,rt_L448),lr_pointer_cast(Int32,match_L412),lr_cast(Int32,Class_L423),lr_pointer_cast(LRLibrary,lib),lr_cast(Int32,(lr_cast(Int32,(-lr_cast(Int32,out_L447))) - lr_cast(Int32,2)))))
            if id_L411==10 || id_L411==13; CUDA.@cuprintf("TRACE1 %d %d %d %d %d %d\n",id_L411,i_L414,out_L447,saved_L421,nodesAdded_L450,matchLevel_L413[rt_L448]);end
            rtLevel_L451=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,matchLevel_L413))[lr_cast(Int32,rt_L448)]))
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,rtLevel_L451) != lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,saved_L421) - lr_cast(Int32,nodesAdded_L450))) < lr_cast(Int32,0)))) || lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,saved_L421) - lr_cast(Int32,nodesAdded_L450))) == lr_cast(Int32,0)))) && lr_truth((!lr_truth(lr_cast(Bool,lr_cast(Int32,fUseZeros))))))))))
                (_lr_old=out_L447; out_L447=lr_cast(Int32,(out_L447+Int32(1))); _lr_old); continue
            end
            if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,saved_L421) - lr_cast(Int32,nodesAdded_L450))) < lr_cast(Int32,reduction_L411)))) || lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,saved_L421) - lr_cast(Int32,nodesAdded_L450))) == lr_cast(Int32,reduction_L411)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,rtLevel_L451) >= lr_cast(Int32,bestLevel_L411)))))))))
                (_lr_old=out_L447; out_L447=lr_cast(Int32,(out_L447+Int32(1))); _lr_old); continue
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,saved_L421) - lr_cast(Int32,nodesAdded_L450)))); reduction_L411 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,rtLevel_L451)); bestLevel_L411 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L414)); bestCut_L411 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,out_L447)); bestOut_L411 = _lr_rhs; _lr_rhs)
            (_lr_old=out_L447; out_L447=lr_cast(Int32,(out_L447+Int32(1))); _lr_old)
        end
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,nleaves_L417))); (lr_pointer_cast(LRCut,cut_L415)).nLeaves = _lr_rhs; _lr_rhs)
        (_lr_old=i_L414; i_L414=lr_cast(Int32,(i_L414+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,bestCut_L411) != lr_cast(Int32,(-lr_cast(Int32,1))))))
        lr_copy!((lr_pointer_cast(LRCut,selectedCuts))[lr_cast(Int32,id_L411)],(lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(id_L411)) * lr_cast(Int32,8))) + lr_cast(Int32,(bestCut_L411)))))],LRCut)
        (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); ((lr_pointer_cast(LRCut,selectedCuts))[lr_cast(Int32,id_L411)]).used = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,bestOut_L411)); (lr_pointer_cast(Int32,bestout))[lr_cast(Int32,id_L411)] = _lr_rhs; _lr_rhs)
    else
        (_lr_rhs=lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,selectedCuts))[lr_cast(Int32,id_L411)]).used = _lr_rhs; _lr_rhs)
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base)))
    return nothing
end

# Original BuildHashTable source device body; candidate, not certified.
@inline function lr_BuildHashTable(ctx,hashTable,sz,fanin0,fanin1,isC0,isC1)::Nothing
    index_L481=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,Int32(blockDim().x)) * lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,index_L481) < lr_cast(Int32,sz))))
        id_L483=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,index_L481) + lr_cast(Int32,1))))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,id_L483)]) == lr_cast(Int32,0)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,id_L483)]) == lr_cast(Int32,0))))))
            begin _lr_result=(nothing; nothing); ; return _lr_result end
        end
        key_L485=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,0)))
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L485 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,id_L483)]) * lr_cast(Int32,7937))))))); key_L485 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L485 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,id_L483)]) * lr_cast(Int32,2971))))))); key_L485 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L485 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,id_L483)]) * lr_cast(Int32,911))))))); key_L485 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(key_L485 ^ lr_cast(UInt64,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,id_L483)]) * lr_cast(Int32,353))))))); key_L485 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,rem(key_L485,lr_cast(UInt64,lr_cast(Int32,ctx.P[0]))))); key_L485 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,id_L483)); ((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(Int32,(lr_cast(Int32,ctx.P[0]) + lr_cast(Int32,index_L481)))]).val = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,1))
            res_L493=lr_cast(Int32,lr_atomicCAS(lr_field_address((lr_pointer_cast(LRTableNode,hashTable))[lr_cast(UInt64,key_L485)],Val(:next)),lr_cast(Int32,(-lr_cast(Int32,1))),lr_cast(Int32,(lr_cast(Int32,ctx.P[0]) + lr_cast(Int32,index_L481)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,res_L493) == lr_cast(Int32,(-lr_cast(Int32,1))))))
                break
            else
                (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,res_L493))); key_L485 = _lr_rhs; _lr_rhs)
            end
        end
    end
    
    return nothing
end

# Original BuildSubgr source device body; candidate, not certified.
@inline function lr_BuildSubgr(ctx,cur,Class,lib,match,isC,fanin0,fanin1,isC0,isC1,phase,newTable,sz)::Nothing
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match))[lr_cast(Int32,cur)]) != lr_cast(Int32,(-lr_cast(Int32,1))))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    in0_L506=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin0))[lr_cast(Int32,Class)]))[lr_cast(Int32,(lr_cast(Int32,cur) - lr_cast(Int32,4)))]))
    in1_L507=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin1))[lr_cast(Int32,Class)]))[lr_cast(Int32,(lr_cast(Int32,cur) - lr_cast(Int32,4)))]))
    lr_BuildSubgr(ctx ,lr_cast(Int32,in0_L506),lr_cast(Int32,Class),lr_pointer_cast(LRLibrary,lib),lr_pointer_cast(Int32,match),lr_cast(UInt64,isC),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,isC0),lr_pointer_cast(Int32,isC1),lr_pointer_cast(Int32,phase),lr_pointer_cast(LRTableNode,newTable),lr_cast(Int32,sz))
    lr_BuildSubgr(ctx ,lr_cast(Int32,in1_L507),lr_cast(Int32,Class),lr_pointer_cast(LRLibrary,lib),lr_pointer_cast(Int32,match),lr_cast(UInt64,isC),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,isC0),lr_pointer_cast(Int32,isC1),lr_pointer_cast(Int32,phase),lr_pointer_cast(LRTableNode,newTable),lr_cast(Int32,sz))
    node_L512=lr_cast(Int32,lr_atomicAdd(ctx.N,lr_cast(Int32,1)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(node_L512 + lr_cast(Int32,1)))); node_L512 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,match))[lr_cast(Int32,in0_L506)])); (lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,match))[lr_cast(Int32,in1_L507)])); (lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt64,(lr_cast(UInt64,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).isC0))[lr_cast(Int32,Class)]))[lr_cast(Int32,(lr_cast(Int32,cur) - lr_cast(Int32,4)))])) ^ (lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,isC) >> lr_cast(Int32,in0_L506))) & lr_cast(UInt64,lr_cast(Int32,1))))))))); (lr_pointer_cast(Int32,isC0))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt64,(lr_cast(UInt64,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).isC1))[lr_cast(Int32,Class)]))[lr_cast(Int32,(lr_cast(Int32,cur) - lr_cast(Int32,4)))])) ^ (lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,isC) >> lr_cast(Int32,in1_L507))) & lr_cast(UInt64,lr_cast(Int32,1))))))))); (lr_pointer_cast(Int32,isC1))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,phase))[lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)])]) ^ lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,node_L512)])))) & (lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,phase))[lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)])]) ^ lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,node_L512)]))))))); (lr_pointer_cast(Int32,phase))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)]) > lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)]))))
        temp_L522=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)]))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)])); (lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L522)); (lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,node_L512)])); temp_L522 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,node_L512)])); (lr_pointer_cast(Int32,isC0))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,temp_L522)); (lr_pointer_cast(Int32,isC1))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
    end
    lr_TableInsert(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)]),lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)]),lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,node_L512)]),lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,node_L512)]),lr_pointer_cast(LRTableNode,newTable),lr_cast(Int32,node_L512),lr_cast(Int32,sz))
    actual_node_L532=lr_cast(Int32,lr_TableLookup(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)]),lr_cast(Int32,(lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)]),lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,node_L512)]),lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,node_L512)]),lr_pointer_cast(LRTableNode,newTable),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,isC0),lr_pointer_cast(Int32,isC1)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,actual_node_L532) != lr_cast(Int32,node_L512))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); (lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs))); (lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,node_L512)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,actual_node_L532)); (lr_pointer_cast(Int32,match))[lr_cast(Int32,cur)] = _lr_rhs; _lr_rhs)
    else
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,node_L512)); (lr_pointer_cast(Int32,match))[lr_cast(Int32,cur)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original ReplaceSubgr source device body; candidate, not certified.
@inline function lr_ReplaceSubgr(ctx,sz,bestout,fanin0,fanin1,isC0,isC1,selectedCuts,lib,hashTable,newTable,phase,replace)::Nothing
    _lr_mem_0=lr_alloc(LRCut)
    cut_L545=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRArray{Int32,54})
    match_L556=lr_pointer_cast(Int32,_lr_mem_1)
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1))))) >= lr_cast(UInt32,lr_cast(Int32,sz)))))
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    end
    id_L544=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,lr_cast(Int32,1)) + lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    lr_copy!(cut_L545,(lr_pointer_cast(LRCut,selectedCuts))[lr_cast(Int32,id_L544)],LRCut)
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(Bool,(cut_L545).used)) == lr_cast(Int32,0)))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,(cut_L545).nLeaves)) < lr_cast(Int32,3))))))
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,(cut_L545).nLeaves)) == lr_cast(Int32,3))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,(cut_L545).leaves))[(_lr_old=(cut_L545).nLeaves; (cut_L545).nLeaves=lr_cast(UInt32,((cut_L545).nLeaves+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    end
    uPhase_L551=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).pPhases))[lr_cast(UInt32,(cut_L545).truthtable)]))
    Class_L552=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).pMap))[lr_cast(UInt32,(cut_L545).truthtable)]))
    pPerm_L553=lr_pointer_cast(Int32,lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,4},(lr_pointer_cast(LRLibrary,lib)).pPerms4))[lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).pPerms))[lr_cast(UInt32,(cut_L545).truthtable)])]))

    isC_L557=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int32,0)))
    j_L558=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L558) < lr_cast(Int32,54))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); (lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,j_L558)] = _lr_rhs; _lr_rhs)
        (_lr_old=j_L558; j_L558=lr_cast(Int32,(j_L558+Int32(1))); _lr_old)
    end
    j_L559=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L559) < lr_cast(Int32,4))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(cut_L545).leaves))[lr_cast(Int32,(lr_pointer_cast(Int32,pPerm_L553))[lr_cast(Int32,j_L559)])])); (lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,j_L559)] = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,uPhase_L551) >> lr_cast(Int32,j_L559))) & lr_cast(Int32,1)))))
            (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(isC_L557 | lr_cast(UInt64,lr_cast(Int64,(lr_cast(Int64,1) << lr_cast(Int32,j_L559))))))); isC_L557 = _lr_rhs; _lr_rhs)
        end
        (_lr_old=j_L559; j_L559=lr_cast(Int32,(j_L559+Int32(1))); _lr_old)
    end
    rt_L564=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,20},(lr_pointer_cast(LRLibrary,lib)).pSubgr))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,(lr_pointer_cast(Int32,bestout))[lr_cast(Int32,id_L544)])]))
    used_L565=lr_cast(UInt64,lr_cast(UInt64,lr_cast(Int64,(lr_cast(Int64,1) << lr_cast(Int32,rt_L564)))))
    j_L566=lr_cast(Int32,lr_cast(Int32,rt_L564))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L566) >= lr_cast(Int32,4))))
        if lr_truth(lr_cast(Bool,lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,used_L565) >> lr_cast(Int32,j_L566))) & lr_cast(UInt64,lr_cast(Int32,1))))))
            (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(used_L565 | lr_cast(UInt64,lr_cast(Int64,(lr_cast(Int64,1) << lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin0))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,(lr_cast(Int32,j_L566) - lr_cast(Int32,4)))]))))))); used_L565 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(used_L565 | lr_cast(UInt64,lr_cast(Int64,(lr_cast(Int64,1) << lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin1))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,(lr_cast(Int32,j_L566) - lr_cast(Int32,4)))]))))))); used_L565 = _lr_rhs; _lr_rhs)
        end
        (_lr_old=j_L566; j_L566=lr_cast(Int32,(j_L566-Int32(1))); _lr_old)
    end
    j_L570=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,j_L570) < lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRLibrary,lib)).nNodes))[lr_cast(Int32,Class_L552)]))))
        if lr_truth(lr_cast(Bool,lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,used_L565) >> (lr_cast(Int32,(lr_cast(Int32,j_L570) + lr_cast(Int32,4)))))) & lr_cast(UInt64,lr_cast(Int32,1))))))
            num_L571=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,j_L570) + lr_cast(Int32,4))))
            in0_L572=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin0))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,j_L570)]))
            in1_L572=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).fanin1))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,j_L570)]))
            if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,in0_L572)]) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,in1_L572)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))))
                (_lr_old=j_L570; j_L570=lr_cast(Int32,(j_L570+Int32(1))); _lr_old); continue
            end
            nodeId_L574=lr_cast(Int32,lr_TableLookup(ctx ,lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,in0_L572)]),lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,in1_L572)]),lr_cast(Int32,lr_cast(UInt64,((lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,isC_L557) >> lr_cast(Int32,in0_L572))) & lr_cast(UInt64,lr_cast(Int32,1))))) ^ lr_cast(UInt64,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).isC0))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,j_L570)]))))),lr_cast(Int32,lr_cast(UInt64,((lr_cast(UInt64,(lr_cast(UInt64,(lr_cast(UInt64,isC_L557) >> lr_cast(Int32,in1_L572))) & lr_cast(UInt64,lr_cast(Int32,1))))) ^ lr_cast(UInt64,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRArray{Int32,60},(lr_pointer_cast(LRLibrary,lib)).isC1))[lr_cast(Int32,Class_L552)]))[lr_cast(Int32,j_L570)]))))),lr_pointer_cast(LRTableNode,hashTable),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,isC0),lr_pointer_cast(Int32,isC1)))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nodeId_L574) != lr_cast(Int32,(-lr_cast(Int32,1))))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nodeId_L574)); (lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,num_L571)] = _lr_rhs; _lr_rhs)
            end
        end
        (_lr_old=j_L570; j_L570=lr_cast(Int32,(j_L570+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,rt_L564)]) == lr_cast(Int32,id_L544))))
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    end
    lr_BuildSubgr(ctx ,lr_cast(Int32,rt_L564),lr_cast(Int32,Class_L552),lr_pointer_cast(LRLibrary,lib),lr_pointer_cast(Int32,match_L556),lr_cast(UInt64,isC_L557),lr_pointer_cast(Int32,fanin0),lr_pointer_cast(Int32,fanin1),lr_pointer_cast(Int32,isC0),lr_pointer_cast(Int32,isC1),lr_pointer_cast(Int32,phase),lr_pointer_cast(LRTableNode,newTable),lr_cast(Int32,sz))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,rt_L564)]) * lr_cast(Int32,2)))) + (lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,phase))[lr_cast(Int32,(lr_pointer_cast(Int32,match_L556))[lr_cast(Int32,rt_L564)])]) ^ lr_cast(Int32,(lr_pointer_cast(Int32,phase))[lr_cast(Int32,id_L544)]))))))); (lr_pointer_cast(Int32,replace))[lr_cast(Int32,id_L544)] = _lr_rhs; _lr_rhs)
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original DetachAndAttach source device body; candidate, not certified.
@inline function lr_DetachAndAttach(ctx,sz,fanin0,fanin1,replace)::Nothing
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1))))) >= lr_cast(UInt32,lr_cast(Int32,sz)))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    id_L589=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,lr_cast(Int32,1)) + lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,replace))[lr_cast(Int32,id_L589)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(Int32,fanin0))[lr_cast(Int32,id_L589)] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,replace))[lr_cast(Int32,id_L589)])); (lr_pointer_cast(Int32,fanin1))[lr_cast(Int32,id_L589)] = _lr_rhs; _lr_rhs)
    
    return nothing
end

# Original Convert source device body; candidate, not certified.
@inline function lr_Convert(ctx,fanin,isC,n)::Nothing
    id_L640=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,id_L640) < lr_cast(Int32,n))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)]) & lr_cast(Int32,1)))); (lr_pointer_cast(Int32,isC))[lr_cast(Int32,id_L640)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)] >> lr_cast(Int32,1)))); (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original Revert source device body; candidate, not certified.
@inline function lr_Revert(ctx,fanin,isC,n)::Nothing
    id_L648=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,id_L648) <= lr_cast(Int32,n))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] << lr_cast(Int32,1)))); (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] + lr_cast(Int32,(lr_pointer_cast(Int32,isC))[lr_cast(Int32,id_L648)])))); (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end
