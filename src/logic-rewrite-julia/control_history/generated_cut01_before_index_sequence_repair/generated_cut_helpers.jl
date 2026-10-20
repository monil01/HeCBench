# GENERATED from original source; full-pipeline verification PENDING.
# Original CutFindValue source device body; candidate, not certified.
@inline function lr_CutFindValue(ctx,cut,nRef)
    value_L80=lr_cast(Int32,lr_cast(Int32,0))
    nOnes_L80=lr_cast(Int32,lr_cast(Int32,0))
    i_L81=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L81) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        (begin value_L80 = lr_cast(Int32,lr_cast(Int32,(value_L80 + lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L81)])])))); value_L80 end)
        (begin nOnes_L80 = lr_cast(Int32,lr_cast(Int32,(nOnes_L80 + lr_cast(Int32,(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L81)])]) == lr_cast(Int32,1)))))))); nOnes_L80 end)
        (begin _lr_old=i_L81; i_L81=lr_cast(Int32,(i_L81+Int32(1))); _lr_old end)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)) < lr_cast(Int32,2))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,1001)); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,value_L80) > lr_cast(Int32,1000))))
        (begin value_L80 = lr_cast(Int32,lr_cast(Int32,1000)); value_L80 end)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nOnes_L80) > lr_cast(Int32,3))))
        (begin value_L80 = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,5) - lr_cast(Int32,nOnes_L80)))); value_L80 end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,value_L80)); ; return _lr_result end
    
    return nothing
end

# Original Inputs source device body; candidate, not certified.
@inline function lr_Inputs(ctx,nRef,cuts,n)
    idx_L94=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1))))) + lr_cast(UInt32,lr_cast(Int32,1))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L94) <= lr_cast(Int32,n))))
        cut_L96=lr_pointer_cast(LRCut,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cuts) + (lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx_L94)) * lr_cast(Int32,8))) + (lr_cast(Int32,0))))))))
        (begin (lr_pointer_cast(LRCut,cut_L96)).used = lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L96)).used end)
        (begin (lr_pointer_cast(LRCut,cut_L96)).sign = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,idx_L94) & lr_cast(Int32,31))))))); (lr_pointer_cast(LRCut,cut_L96)).sign end)
        (begin (lr_pointer_cast(LRCut,cut_L96)).truthtable = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,43690))); (lr_pointer_cast(LRCut,cut_L96)).truthtable end)
        (begin (lr_pointer_cast(LRCut,cut_L96)).nLeaves = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L96)).nLeaves end)
        (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L96)).leaves))[lr_cast(Int32,0)] = lr_cast(Int32,lr_cast(Int32,idx_L94)); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L96)).leaves))[lr_cast(Int32,0)] end)
        (begin (lr_pointer_cast(LRCut,cut_L96)).value = lr_cast(UInt32,lr_cast(UInt32,lr_CutFindValue(ctx ,lr_pointer_cast(LRCut,cut_L96),lr_pointer_cast(Int32,nRef)))); (lr_pointer_cast(LRCut,cut_L96)).value end)
    end
    
    return nothing
end

# Original CountOnes source device body; candidate, not certified.
@inline function lr_CountOnes(ctx,uWord)
    (begin uWord = lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,1431655765))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,1)))) & lr_cast(UInt32,lr_cast(Int32,1431655765)))))))); uWord end)
    (begin uWord = lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,858993459))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,2)))) & lr_cast(UInt32,lr_cast(Int32,858993459)))))))); uWord end)
    (begin uWord = lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,252645135))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,4)))) & lr_cast(UInt32,lr_cast(Int32,252645135)))))))); uWord end)
    (begin uWord = lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,16711935))))) + (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,8)))) & lr_cast(UInt32,lr_cast(Int32,16711935)))))))); uWord end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uWord) & lr_cast(UInt32,lr_cast(Int32,65535))))) + (lr_cast(UInt32,(lr_cast(UInt32,uWord) >> lr_cast(Int32,16)))))))); ; return _lr_result end
    
    return nothing
end

# Original FindCut source device body; candidate, not certified.
@inline function lr_FindCut(ctx,idx,cuts)
    i_L115=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L115) < lr_cast(Int32,8))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(Bool,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L115)))))]).used)) == lr_cast(Int32,0))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,i_L115)); ; return _lr_result end
        end
        (begin _lr_old=i_L115; i_L115=lr_cast(Int32,(i_L115+Int32(1))); _lr_old end)
    end
    ans_L117=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    i_L118=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L118) < lr_cast(Int32,8))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L118)))))]).nLeaves)) > lr_cast(Int32,2))))
            if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L118)))))]).value)) < lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).value)))))))
                (begin ans_L117 = lr_cast(Int32,lr_cast(Int32,i_L118)); ans_L117 end)
            end
        end
        (begin _lr_old=i_L118; i_L118=lr_cast(Int32,(i_L118+Int32(1))); _lr_old end)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        i_L124=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,8))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L124)))))]).nLeaves)) == lr_cast(Int32,2))))
                if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L124)))))]).value)) < lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).value)))))))
                    (begin ans_L117 = lr_cast(Int32,lr_cast(Int32,i_L124)); ans_L117 end)
                end
            end
            (begin _lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old end)
        end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        i_L131=lr_cast(Int32,lr_cast(Int32,0))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L131) < lr_cast(Int32,8))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L131)))))]).nLeaves)) < lr_cast(Int32,2))))
                if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,ans_L117) == lr_cast(Int32,(-lr_cast(Int32,1)))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(i_L131)))))]).value)) < lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).value)))))))
                    (begin ans_L117 = lr_cast(Int32,lr_cast(Int32,i_L131)); ans_L117 end)
                end
            end
            (begin _lr_old=i_L131; i_L131=lr_cast(Int32,(i_L131+Int32(1))); _lr_old end)
        end
    end
    (begin ((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).used = lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,cuts))[(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx)) * lr_cast(Int32,8))) + lr_cast(Int32,(ans_L117)))))]).used end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,ans_L117)); ; return _lr_result end
    
    return nothing
end

# Original MergeCutOrdered source device body; candidate, not certified.
@inline function lr_MergeCutOrdered(ctx,a,b,cut)
    _lr_mem_0=lr_alloc(LRCut)
    lr_copy!(_lr_mem_0,a,LRCut)
    a=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    lr_copy!(_lr_mem_1,b,LRCut)
    b=_lr_mem_1[0]
    c_L143=lr_cast(Int32,lr_cast(Int32,0))
    i_L143=lr_cast(Int32,lr_cast(Int32,0))
    k_L143=lr_cast(Int32,lr_cast(Int32,0))
    (begin c_L143 = lr_cast(Int32,lr_cast(Int32,0)); c_L143 end)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,c_L143) < lr_cast(Int32,4))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L143) == lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L143) == lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)))))
                (begin (lr_pointer_cast(LRCut,cut)).nLeaves = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,c_L143))); (lr_pointer_cast(LRCut,cut)).nLeaves end)
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
            end
            (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[(begin _lr_old=i_L143; i_L143=lr_cast(Int32,(i_L143+Int32(1))); _lr_old end)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] end)
            (begin _lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old end); continue
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L143) == lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L143) == lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))
                (begin (lr_pointer_cast(LRCut,cut)).nLeaves = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,c_L143))); (lr_pointer_cast(LRCut,cut)).nLeaves end)
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
            end
            (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[(begin _lr_old=k_L143; k_L143=lr_cast(Int32,(k_L143+Int32(1))); _lr_old end)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] end)
            (begin _lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old end); continue
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[lr_cast(Int32,i_L143)]) < lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[lr_cast(Int32,k_L143)]))))
            (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[(begin _lr_old=i_L143; i_L143=lr_cast(Int32,(i_L143+Int32(1))); _lr_old end)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] end)
            (begin _lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old end); continue
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[lr_cast(Int32,i_L143)]) > lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[lr_cast(Int32,k_L143)]))))
            (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(b).leaves))[(begin _lr_old=k_L143; k_L143=lr_cast(Int32,(k_L143+Int32(1))); _lr_old end)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] end)
            (begin _lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old end); continue
        end
        (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] = lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(a).leaves))[(begin _lr_old=i_L143; i_L143=lr_cast(Int32,(i_L143+Int32(1))); _lr_old end)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,c_L143)] end)
        (begin _lr_old=k_L143; k_L143=lr_cast(Int32,(k_L143+Int32(1))); _lr_old end)
        (begin _lr_old=c_L143; c_L143=lr_cast(Int32,(c_L143+Int32(1))); _lr_old end)
    end
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L143) < lr_cast(Int32,lr_cast(UInt32,(a).nLeaves))))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L143) < lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)))))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    end
    (begin (lr_pointer_cast(LRCut,cut)).nLeaves = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,c_L143))); (lr_pointer_cast(LRCut,cut)).nLeaves end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original MergeCut source device body; candidate, not certified.
@inline function lr_MergeCut(ctx,a,b,cut)
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
    (begin (lr_pointer_cast(LRCut,cut)).sign = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(a).sign) | lr_cast(Int32,(b).sign)))); (lr_pointer_cast(LRCut,cut)).sign end)
    (begin (lr_pointer_cast(LRCut,cut)).used = lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut)).used end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original CutDominance source device body; candidate, not certified.
@inline function lr_CutDominance(ctx,a,b)
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
                (begin ok_L199 = lr_cast(Int32,lr_cast(Int32,1)); ok_L199 end)
            end
            (begin _lr_old=j_L200; j_L200=lr_cast(Int32,(j_L200+Int32(1))); _lr_old end)
        end
        if lr_truth((!lr_truth(lr_cast(Bool,lr_cast(Int32,ok_L199)))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
        end
        (begin _lr_old=i_L198; i_L198=lr_cast(Int32,(i_L198+Int32(1))); _lr_old end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original CutFilter source device body; candidate, not certified.
@inline function lr_CutFilter(ctx,cut,id)
    i_L209=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L209) < lr_cast(Int32,8))))
        if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L209) != lr_cast(Int32,id)))) && lr_truth(lr_cast(Bool,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).used))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).nLeaves)) > lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).nLeaves)))))
                if lr_truth(lr_cast(Bool,((lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).sign) & lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).sign)))) != lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).sign))))
                    (begin _lr_old=i_L209; i_L209=lr_cast(Int32,(i_L209+Int32(1))); _lr_old end); continue
                end
                if lr_truth(lr_cast(Bool,lr_CutDominance(ctx ,(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)],(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)])))
                    (begin ((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).used = lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).used end)
                end
            else
                if lr_truth(lr_cast(Bool,((lr_cast(Int32,(lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).sign) & lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).sign)))) != lr_cast(Int32,((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)]).sign))))
                    (begin _lr_old=i_L209; i_L209=lr_cast(Int32,(i_L209+Int32(1))); _lr_old end); continue
                end
                if lr_truth(lr_cast(Bool,lr_CutDominance(ctx ,(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,i_L209)],(lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)])))
                    (begin ((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).used = lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,0))); ((lr_pointer_cast(LRCut,cut))[lr_cast(Int32,id)]).used end)
                    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
                end
            end
        end
        (begin _lr_old=i_L209; i_L209=lr_cast(Int32,(i_L209+Int32(1))); _lr_old end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    
    return nothing
end

# Original CutTruthPhase source device body; candidate, not certified.
@inline function lr_CutTruthPhase(ctx,x,cut)
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
                (begin phase_L226 = lr_cast(Int32,lr_cast(Int32,(phase_L226 | lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L227)))))); phase_L226 end)
            end
            (begin _lr_old=j_L228; j_L228=lr_cast(Int32,(j_L228+Int32(1))); _lr_old end)
        end
        (begin _lr_old=i_L227; i_L227=lr_cast(Int32,(i_L227+Int32(1))); _lr_old end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,phase_L226)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original CutTruthSwapAdjacentVars source device body; candidate, not certified.
@inline function lr_CutTruthSwapAdjacentVars(ctx,uTruth,iVar)
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
@inline function lr_CutTruthStretch(ctx,truthtable,nVar,phase)
    Var_L244=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,nVar) - lr_cast(Int32,1))))
    i_L245=lr_cast(Int32,lr_cast(Int32,3))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L245) >= lr_cast(Int32,0))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,phase) >> lr_cast(Int32,i_L245))) & lr_cast(Int32,1)))))
            k_L246=lr_cast(Int32,lr_cast(Int32,Var_L244))
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L246) < lr_cast(Int32,i_L245))))
                (begin truthtable = lr_cast(Int32,lr_CutTruthSwapAdjacentVars(ctx ,lr_cast(Int32,truthtable),lr_cast(Int32,k_L246))); truthtable end)
                (begin _lr_old=k_L246; k_L246=lr_cast(Int32,(k_L246+Int32(1))); _lr_old end)
            end
            (begin _lr_old=Var_L244; Var_L244=lr_cast(Int32,(Var_L244-Int32(1))); _lr_old end)
        end
        (begin _lr_old=i_L245; i_L245=lr_cast(Int32,(i_L245-Int32(1))); _lr_old end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,truthtable)); ; return _lr_result end
    
    return nothing
end

# Original CutTruthtable source device body; candidate, not certified.
@inline function lr_CutTruthtable(ctx,cut,a,b,aComplement,bComplement)
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
    (begin tt0_L254 = lr_cast(Int32,lr_CutTruthStretch(ctx ,lr_cast(Int32,tt0_L254),lr_cast(Int32,lr_cast(UInt32,(a).nLeaves)),lr_CutTruthPhase(ctx ,a,cut))); tt0_L254 end)
    (begin tt1_L255 = lr_cast(Int32,lr_CutTruthStretch(ctx ,lr_cast(Int32,tt1_L255),lr_cast(Int32,lr_cast(UInt32,(b).nLeaves)),lr_CutTruthPhase(ctx ,b,cut))); tt1_L255 end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,tt0_L254) & lr_cast(Int32,tt1_L255)))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base)))
    return nothing
end

# Original CutTruthShrink source device body; candidate, not certified.
@inline function lr_CutTruthShrink(ctx,uTruth,nVars,Phase)
    local i_L262
    local k_L262
    Var_L262=lr_cast(Int32,lr_cast(Int32,0))
    (begin i_L262 = lr_cast(Int32,lr_cast(Int32,0)); i_L262 end)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L262) < lr_cast(Int32,4))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,Phase) & (lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L262))))))))
            (begin k_L262 = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,i_L262) - lr_cast(Int32,1)))); k_L262 end)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L262) >= lr_cast(Int32,Var_L262))))
                (begin uTruth = lr_cast(Int32,lr_CutTruthSwapAdjacentVars(ctx ,lr_cast(Int32,uTruth),lr_cast(Int32,k_L262))); uTruth end)
                (begin _lr_old=k_L262; k_L262=lr_cast(Int32,(k_L262-Int32(1))); _lr_old end)
            end
            (begin _lr_old=Var_L262; Var_L262=lr_cast(Int32,(Var_L262+Int32(1))); _lr_old end)
        end
        (begin _lr_old=i_L262; i_L262=lr_cast(Int32,(i_L262+Int32(1))); _lr_old end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,uTruth)); ; return _lr_result end
    
    return nothing
end

# Original MinimizeCutSupport source device body; candidate, not certified.
@inline function lr_MinimizeCutSupport(ctx,cut)
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
            (begin _lr_old=nLeaves_L280; nLeaves_L280=lr_cast(Int32,(nLeaves_L280-Int32(1))); _lr_old end)
        else
            (begin phase_L280 = lr_cast(Int32,lr_cast(Int32,(phase_L280 | lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i_L281)))))); phase_L280 end)
        end
        (begin _lr_old=i_L281; i_L281=lr_cast(Int32,(i_L281+Int32(1))); _lr_old end)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLeaves_L280) == lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
    end
    (begin truth_L280 = lr_cast(Int32,lr_CutTruthShrink(ctx ,lr_cast(Int32,truth_L280),lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)),lr_cast(Int32,phase_L280))); truth_L280 end)
    (begin (lr_pointer_cast(LRCut,cut)).truthtable = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(lr_cast(Int32,truth_L280) & lr_cast(Int32,65535))))); (lr_pointer_cast(LRCut,cut)).truthtable end)
    (begin (lr_pointer_cast(LRCut,cut)).sign = lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRCut,cut)).sign end)
    i_L290=lr_cast(Int32,lr_cast(Int32,0))
    k_L290=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L290) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,phase_L280) >> lr_cast(Int32,i_L290))) & lr_cast(Int32,1)))))
            (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[(begin _lr_old=k_L290; k_L290=lr_cast(Int32,(k_L290+Int32(1))); _lr_old end)] = lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L290)])); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[(begin _lr_old=k_L290; k_L290=lr_cast(Int32,(k_L290+Int32(1))); _lr_old end)] end)
            (begin (lr_pointer_cast(LRCut,cut)).sign = lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(LRCut,cut)).sign | lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,31) & lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L290)]))))))))); (lr_pointer_cast(LRCut,cut)).sign end)
        end
        (begin _lr_old=i_L290; i_L290=lr_cast(Int32,(i_L290+Int32(1))); _lr_old end)
    end
    (begin (lr_pointer_cast(LRCut,cut)).nLeaves = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,nLeaves_L280))); (lr_pointer_cast(LRCut,cut)).nLeaves end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end

# Original CutEnumerate source device body; candidate, not certified.
@inline function lr_CutEnumerate(ctx,fanin0,fanin1,isC0,isC1,nRef,cuts,delta,n)
    _lr_mem_0=lr_alloc(LRCut)
    a_L312=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRCut)
    b_L312=_lr_mem_1[0]
    idx_L299=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L299) < lr_cast(Int32,n))))
        (begin idx_L299 = lr_cast(Int32,lr_cast(Int32,(idx_L299 + lr_cast(Int32,(lr_cast(Int32,delta) + lr_cast(Int32,1)))))); idx_L299 end)
        cut_L302=lr_pointer_cast(LRCut,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cuts) + (lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(idx_L299)) * lr_cast(Int32,8))) + (lr_cast(Int32,0))))))))
        (begin (lr_pointer_cast(LRCut,cut_L302)).used = lr_cast(Bool,lr_cast(Bool,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L302)).used end)
        (begin (lr_pointer_cast(LRCut,cut_L302)).sign = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,idx_L299) & lr_cast(Int32,31))))))); (lr_pointer_cast(LRCut,cut_L302)).sign end)
        (begin (lr_pointer_cast(LRCut,cut_L302)).truthtable = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,43690))); (lr_pointer_cast(LRCut,cut_L302)).truthtable end)
        (begin (lr_pointer_cast(LRCut,cut_L302)).nLeaves = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,1))); (lr_pointer_cast(LRCut,cut_L302)).nLeaves end)
        (begin (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L302)).leaves))[lr_cast(Int32,0)] = lr_cast(Int32,lr_cast(Int32,idx_L299)); (lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut_L302)).leaves))[lr_cast(Int32,0)] end)
        (begin (lr_pointer_cast(LRCut,cut_L302)).value = lr_cast(UInt32,lr_cast(UInt32,lr_CutFindValue(ctx ,lr_pointer_cast(LRCut,cut_L302),lr_pointer_cast(Int32,nRef)))); (lr_pointer_cast(LRCut,cut_L302)).value end)
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
                            (begin _lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old end); continue
                        end
                        cutId_L314=lr_cast(Int32,lr_FindCut(ctx ,lr_cast(Int32,idx_L299),lr_pointer_cast(LRCut,cuts)))
                        if lr_truth((!lr_truth(lr_cast(Bool,lr_MergeCut(ctx ,a_L312,b_L312,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cut_L302) + lr_cast(Int32,cutId_L314))))))))
                            (begin _lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old end); continue
                        end
                        if lr_truth(lr_cast(Bool,lr_CutFilter(ctx ,lr_pointer_cast(LRCut,cut_L302),lr_cast(Int32,cutId_L314))))
                            (begin _lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old end); continue
                        end
                        (begin ((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).truthtable = lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(lr_cast(Int32,65535) & lr_CutTruthtable(ctx ,(lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)],a_L312,b_L312,lr_cast(Int32,(lr_pointer_cast(Int32,isC0))[lr_cast(Int32,idx_L299)]),lr_cast(Int32,(lr_pointer_cast(Int32,isC1))[lr_cast(Int32,idx_L299)])))))); ((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).truthtable end)
                        if lr_truth(lr_cast(Bool,lr_MinimizeCutSupport(ctx ,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cut_L302) + lr_cast(Int32,cutId_L314))))))
                            lr_CutFilter(ctx ,lr_pointer_cast(LRCut,cut_L302),lr_cast(Int32,cutId_L314))
                        end
                        (begin ((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).value = lr_cast(UInt32,lr_cast(UInt32,lr_CutFindValue(ctx ,lr_pointer_cast(LRCut,(lr_pointer_cast(LRCut,cut_L302) + lr_cast(Int32,cutId_L314))),lr_pointer_cast(Int32,nRef)))); ((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).value end)
                        if lr_truth(lr_cast(Bool,(lr_cast(Int32,lr_cast(UInt32,((lr_pointer_cast(LRCut,cut_L302))[lr_cast(Int32,cutId_L314)]).nLeaves)) < lr_cast(Int32,2))))
                            begin _lr_result=(begin nothing; nothing end); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); return _lr_result end
                        end
                    end
                    (begin _lr_old=j_L311; j_L311=lr_cast(Int32,(j_L311+Int32(1))); _lr_old end)
                end
            end
            (begin _lr_old=i_L310; i_L310=lr_cast(Int32,(i_L310+Int32(1))); _lr_old end)
        end
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end

# Original Decrease source device body; candidate, not certified.
@inline function lr_Decrease(ctx,id,tableSize,tableId,tableNum,nRef,cut)
    i_L327=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L327) < lr_cast(Int32,lr_cast(UInt32,(lr_pointer_cast(LRCut,cut)).nLeaves)))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,id) == lr_cast(Int32,(lr_pointer_cast(Int32,(lr_pointer_cast(LRCut,cut)).leaves))[lr_cast(Int32,i_L327)]))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
        end
        (begin _lr_old=i_L327; i_L327=lr_cast(Int32,(i_L327+Int32(1))); _lr_old end)
    end
    i_L329=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L329) < lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableId))[lr_cast(Int32,i_L329)]) == lr_cast(Int32,id))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,(begin (lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L329)]=lr_cast(Int32,((lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L329)]-Int32(1))); (lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L329)] end))); ; return _lr_result end
        end
        (begin _lr_old=i_L329; i_L329=lr_cast(Int32,(i_L329+Int32(1))); _lr_old end)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]) == lr_cast(Int32,8))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    end
    (begin (lr_pointer_cast(Int32,tableId))[lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)])] = lr_cast(Int32,lr_cast(Int32,id)); (lr_pointer_cast(Int32,tableId))[lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)])] end)
    (begin (lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)])] = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,id)]) - lr_cast(Int32,1)))); (lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)])] end)
    (begin _lr_old=(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]; (lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]=lr_cast(Int32,((lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]+Int32(1))); _lr_old end)
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,nRef))[lr_cast(Int32,id)]) - lr_cast(Int32,1)))); ; return _lr_result end
    
    return nothing
end

# Original IsDeleted source device body; candidate, not certified.
@inline function lr_IsDeleted(ctx,id,tableSize,tableId,tableNum)
    i_L343=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L343) < lr_cast(Int32,(lr_pointer_cast(Int32,tableSize))[lr_cast(Int32,0)]))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableId))[lr_cast(Int32,i_L343)]) == lr_cast(Int32,id))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,tableNum))[lr_cast(Int32,i_L343)]) == lr_cast(Int32,0))))); ; return _lr_result end
        end
        (begin _lr_old=i_L343; i_L343=lr_cast(Int32,(i_L343+Int32(1))); _lr_old end)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    
    return nothing
end

# Original Convert source device body; candidate, not certified.
@inline function lr_Convert(ctx,fanin,isC,n)
    id_L640=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,id_L640) < lr_cast(Int32,n))))
        (begin (lr_pointer_cast(Int32,isC))[lr_cast(Int32,id_L640)] = lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)]) & lr_cast(Int32,1)))); (lr_pointer_cast(Int32,isC))[lr_cast(Int32,id_L640)] end)
        (begin (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)] = lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)] >> lr_cast(Int32,1)))); (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L640)] end)
    end
    
    return nothing
end

# Original Revert source device body; candidate, not certified.
@inline function lr_Revert(ctx,fanin,isC,n)
    id_L648=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,id_L648) <= lr_cast(Int32,n))))
        (begin (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] = lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] << lr_cast(Int32,1)))); (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] end)
        (begin (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] = lr_cast(Int32,lr_cast(Int32,((lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] + lr_cast(Int32,(lr_pointer_cast(Int32,isC))[lr_cast(Int32,id_L648)])))); (lr_pointer_cast(Int32,fanin))[lr_cast(Int32,id_L648)] end)
    end
    
    return nothing
end
