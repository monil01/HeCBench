# GENERATED from original source; full-pipeline verification PENDING.
# Original isConst0 source device body; candidate, not certified.
@inline function lr_isConst0(ctx,pIn,nVars)::Int32
    local w_L9
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L9 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L9) >= lr_cast(Int32,0))))
        if lr_truth(lr_cast(Bool,lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn))[lr_cast(Int32,w_L9)])))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
        end
        (_lr_old=w_L9; w_L9=lr_cast(Int32,(w_L9-Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    
    return nothing
end

# Original isConst1 source device body; candidate, not certified.
@inline function lr_isConst1(ctx,pIn,nVars)::Int32
    local w_L18
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L18 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L18) >= lr_cast(Int32,0))))
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn))[lr_cast(Int32,w_L18)]) != lr_cast(UInt32,(~lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))))))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
        end
        (_lr_old=w_L18; w_L18=lr_cast(Int32,(w_L18-Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    
    return nothing
end

# Original clear source device body; candidate, not certified.
@inline function lr_clear(ctx,pOut,nVars)::Nothing
    local w_L27
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L27 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L27) >= lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); (lr_pointer_cast(UInt32,pOut))[lr_cast(Int32,w_L27)] = _lr_rhs; _lr_rhs)
        (_lr_old=w_L27; w_L27=lr_cast(Int32,(w_L27-Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original fill source device body; candidate, not certified.
@inline function lr_fill(ctx,pOut,nVars)::Nothing
    local w_L34
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L34 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L34) >= lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(~lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0)))))); (lr_pointer_cast(UInt32,pOut))[lr_cast(Int32,w_L34)] = _lr_rhs; _lr_rhs)
        (_lr_old=w_L34; w_L34=lr_cast(Int32,(w_L34-Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original truthAnd source device body; candidate, not certified.
@inline function lr_truthAnd(ctx,pOut,pIn0,pIn1,nVars)::Nothing
    local w_L41
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L41 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L41) >= lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn0))[lr_cast(Int32,w_L41)]) & lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn1))[lr_cast(Int32,w_L41)])))); (lr_pointer_cast(UInt32,pOut))[lr_cast(Int32,w_L41)] = _lr_rhs; _lr_rhs)
        (_lr_old=w_L41; w_L41=lr_cast(Int32,(w_L41-Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original truthOr source device body; candidate, not certified.
@inline function lr_truthOr(ctx,pOut,pIn0,pIn1,nVars)::Nothing
    local w_L48
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L48 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L48) >= lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn0))[lr_cast(Int32,w_L48)]) | lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn1))[lr_cast(Int32,w_L48)])))); (lr_pointer_cast(UInt32,pOut))[lr_cast(Int32,w_L48)] = _lr_rhs; _lr_rhs)
        (_lr_old=w_L48; w_L48=lr_cast(Int32,(w_L48-Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original truthNot source device body; candidate, not certified.
@inline function lr_truthNot(ctx,pOut,pIn,nVars)::Nothing
    local w_L55
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L55 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L55) >= lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn))[lr_cast(Int32,w_L55)])))); (lr_pointer_cast(UInt32,pOut))[lr_cast(Int32,w_L55)] = _lr_rhs; _lr_rhs)
        (_lr_old=w_L55; w_L55=lr_cast(Int32,(w_L55-Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original truthSharp source device body; candidate, not certified.
@inline function lr_truthSharp(ctx,pOut,pIn0,pIn1,nVars)::Nothing
    local w_L62
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L62 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L62) >= lr_cast(Int32,0))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn0))[lr_cast(Int32,w_L62)]) & lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn1))[lr_cast(Int32,w_L62)])))))); (lr_pointer_cast(UInt32,pOut))[lr_cast(Int32,w_L62)] = _lr_rhs; _lr_rhs)
        (_lr_old=w_L62; w_L62=lr_cast(Int32,(w_L62-Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original truthEqual source device body; candidate, not certified.
@inline function lr_truthEqual(ctx,pIn0,pIn1,nVars)::Int32
    local w_L69
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_factor_TruthWordNum(lr_cast(Int32,nVars)) - lr_cast(Int32,1)))); w_L69 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L69) >= lr_cast(Int32,0))))
        if lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn0))[lr_cast(Int32,w_L69)]) != lr_cast(UInt32,(lr_pointer_cast(UInt32,pIn1))[lr_cast(Int32,w_L69)]))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
        end
        (_lr_old=w_L69; w_L69=lr_cast(Int32,(w_L69-Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    
    return nothing
end

# Original varInSupport source device body; candidate, not certified.
@inline function lr_varInSupport(ctx,pTruth,nVars,iVar)::Int32
    nWords_L79=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVars)))
    local i_L80
    local k_L80
    local Step_L80
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iVar) < lr_cast(Int32,nVars))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_cast(Int32,iVar)==lr_cast(Int32,0)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L80 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L80) < lr_cast(Int32,nWords_L79))))
            if lr_truth(lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,lr_cast(Int32,1431655765))))) != (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,2863311530)))) >> lr_cast(Int32,1)))))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
            end
            (_lr_old=i_L80; i_L80=lr_cast(Int32,(i_L80+Int32(1))); _lr_old)
        end
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,1)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L80 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L80) < lr_cast(Int32,nWords_L79))))
            if lr_truth(lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,lr_cast(Int32,858993459))))) != (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,3435973836)))) >> lr_cast(Int32,2)))))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
            end
            (_lr_old=i_L80; i_L80=lr_cast(Int32,(i_L80+Int32(1))); _lr_old)
        end
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,2)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L80 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L80) < lr_cast(Int32,nWords_L79))))
            if lr_truth(lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,lr_cast(Int32,252645135))))) != (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,4042322160)))) >> lr_cast(Int32,4)))))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
            end
            (_lr_old=i_L80; i_L80=lr_cast(Int32,(i_L80+Int32(1))); _lr_old)
        end
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,3)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L80 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L80) < lr_cast(Int32,nWords_L79))))
            if lr_truth(lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,lr_cast(Int32,16711935))))) != (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,4278255360)))) >> lr_cast(Int32,8)))))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
            end
            (_lr_old=i_L80; i_L80=lr_cast(Int32,(i_L80+Int32(1))); _lr_old)
        end
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,4)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L80 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L80) < lr_cast(Int32,nWords_L79))))
            if lr_truth(lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,lr_cast(Int32,65535))))) != (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) & lr_cast(UInt32,4294901760)))) >> lr_cast(Int32,16)))))))
                begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
            end
            (_lr_old=i_L80; i_L80=lr_cast(Int32,(i_L80+Int32(1))); _lr_old)
        end
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    else
        (_lr_rhs=lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,iVar) - lr_cast(Int32,5)))))))); Step_L80 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L80 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L80) < lr_cast(Int32,nWords_L79))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L80 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L80) < lr_cast(Int32,Step_L80))))
                if lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L80)]) != lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,(lr_cast(Int32,Step_L80) + lr_cast(Int32,i_L80)))]))))
                    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
                end
                (_lr_old=i_L80; i_L80=lr_cast(Int32,(i_L80+Int32(1))); _lr_old)
            end
            (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(pTruth + lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,Step_L80)))))); pTruth = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(k_L80 + lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,Step_L80)))))); k_L80 = _lr_rhs; _lr_rhs)
        end
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    
    return nothing
end

# Original cofactor0 source device body; candidate, not certified.
@inline function lr_cofactor0(ctx,pTruth,nVars,iVar)::Nothing
    nWords_L123=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVars)))
    local i_L124
    local k_L124
    local Step_L124
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iVar) < lr_cast(Int32,nVars))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_cast(Int32,iVar)==lr_cast(Int32,0)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L124 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,nWords_L123))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,1431655765))))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,1431655765))))) << lr_cast(Int32,1))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,1)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L124 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,nWords_L123))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,858993459))))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,858993459))))) << lr_cast(Int32,2))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,2)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L124 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,nWords_L123))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,252645135))))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,252645135))))) << lr_cast(Int32,4))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,3)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L124 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,nWords_L123))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,16711935))))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,16711935))))) << lr_cast(Int32,8))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,4)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L124 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,nWords_L123))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,65535))))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)]) & lr_cast(UInt32,lr_cast(Int32,65535))))) << lr_cast(Int32,16))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    else
        (_lr_rhs=lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,iVar) - lr_cast(Int32,5)))))))); Step_L124 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L124 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L124) < lr_cast(Int32,nWords_L123))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L124 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L124) < lr_cast(Int32,Step_L124))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L124)])); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,(lr_cast(Int32,Step_L124) + lr_cast(Int32,i_L124)))] = _lr_rhs; _lr_rhs)
                (_lr_old=i_L124; i_L124=lr_cast(Int32,(i_L124+Int32(1))); _lr_old)
            end
            (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(pTruth + lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,Step_L124)))))); pTruth = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(k_L124 + lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,Step_L124)))))); k_L124 = _lr_rhs; _lr_rhs)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    
    return nothing
end

# Original cofactor1 source device body; candidate, not certified.
@inline function lr_cofactor1(ctx,pTruth,nVars,iVar)::Nothing
    nWords_L161=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVars)))
    local i_L162
    local k_L162
    local Step_L162
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iVar) < lr_cast(Int32,nVars))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_cast(Int32,iVar)==lr_cast(Int32,0)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L162 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L162) < lr_cast(Int32,nWords_L161))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,2863311530)))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,2863311530)))) >> lr_cast(Int32,1))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L162; i_L162=lr_cast(Int32,(i_L162+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,1)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L162 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L162) < lr_cast(Int32,nWords_L161))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,3435973836)))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,3435973836)))) >> lr_cast(Int32,2))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L162; i_L162=lr_cast(Int32,(i_L162+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,2)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L162 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L162) < lr_cast(Int32,nWords_L161))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,4042322160)))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,4042322160)))) >> lr_cast(Int32,4))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L162; i_L162=lr_cast(Int32,(i_L162+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,3)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L162 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L162) < lr_cast(Int32,nWords_L161))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,4278255360)))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,4278255360)))) >> lr_cast(Int32,8))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L162; i_L162=lr_cast(Int32,(i_L162+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    elseif lr_cast(Int32,iVar)==lr_cast(Int32,4)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L162 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L162) < lr_cast(Int32,nWords_L161))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,4294901760)))) | (lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)]) & lr_cast(UInt32,4294901760)))) >> lr_cast(Int32,16))))))); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L162; i_L162=lr_cast(Int32,(i_L162+Int32(1))); _lr_old)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    else
        (_lr_rhs=lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,(lr_cast(Int32,iVar) - lr_cast(Int32,5)))))))); Step_L162 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L162 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L162) < lr_cast(Int32,nWords_L161))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L162 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L162) < lr_cast(Int32,Step_L162))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,(lr_cast(Int32,Step_L162) + lr_cast(Int32,i_L162)))])); (lr_pointer_cast(UInt32,pTruth))[lr_cast(Int32,i_L162)] = _lr_rhs; _lr_rhs)
                (_lr_old=i_L162; i_L162=lr_cast(Int32,(i_L162+Int32(1))); _lr_old)
            end
            (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(pTruth + lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,Step_L162)))))); pTruth = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(k_L162 + lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,Step_L162)))))); k_L162 = _lr_rhs; _lr_rhs)
        end
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    
    return nothing
end

# Original stretch source device body; candidate, not certified.
@inline function lr_stretch(ctx,pInOut,nVarS,nVarB)::Nothing
    local w_L199
    local i_L199
    local step_L199
    local nWords_L199
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nVarS) == lr_cast(Int32,nVarB))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,nVarS) < lr_cast(Int32,nVarB))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVarS))); step_L199 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVarB))); nWords_L199 = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,step_L199) == lr_cast(Int32,nWords_L199))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,step_L199) < lr_cast(Int32,nWords_L199))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); w_L199 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,w_L199) < lr_cast(Int32,nWords_L199))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L199 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L199) < lr_cast(Int32,step_L199))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,pInOut))[lr_cast(Int32,i_L199)])); (lr_pointer_cast(UInt32,pInOut))[lr_cast(Int32,(lr_cast(Int32,w_L199) + lr_cast(Int32,i_L199)))] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L199; i_L199=lr_cast(Int32,(i_L199+Int32(1))); _lr_old)
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(w_L199 + lr_cast(Int32,step_L199)))); w_L199 = _lr_rhs; _lr_rhs)
    end
    
    return nothing
end

# Original unbindAndNodeKeyFlag source device body; candidate, not certified.
@inline function lr_unbindAndNodeKeyFlag(ctx,key,lit1,lit2,pfComp)::Nothing
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,(lr_cast(UInt64,(lr_cast(UInt64,key) & lr_cast(UInt64,lr_cast(UInt64,1)))))))); (lr_pointer_cast(Int32,pfComp))[0] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,(lr_cast(UInt64,((lr_cast(UInt64,(lr_cast(UInt64,key) & lr_cast(UInt64,lr_cast(UInt64,4294967295))))) >> lr_cast(Int32,1))))))); (lr_pointer_cast(Int32,lit2))[0] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,lr_cast(Int32,(lr_cast(UInt64,(lr_cast(UInt64,key) >> lr_cast(Int32,32))))))); (lr_pointer_cast(Int32,lit1))[0] = _lr_rhs; _lr_rhs)
    
    return nothing
end

# Original formAndNodeKeyFlag source device body; candidate, not certified.
@inline function lr_formAndNodeKeyFlag(ctx,lit1,lit2,fComp)::UInt64
    ((lr_truth(lr_cast(Bool,(lr_truth((lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,lit1) >= lr_cast(Int32,0)))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,lit2) >= lr_cast(Int32,0)))))) && lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,fComp) == lr_cast(Int32,0)))) || lr_truth(lr_cast(Bool,(lr_cast(Int32,fComp) == lr_cast(Int32,1))))))))) && lr_truth(lr_cast(Bool,(lr_cast(Int32,lit1) <= lr_cast(Int32,lit2))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    uLit1_L36=lr_cast(UInt32,lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,lit1))))
    uLit2_L37=lr_cast(UInt32,lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,lit2))))
    key_L38=lr_cast(UInt64,lr_cast(UInt64,(lr_cast(UInt64,((lr_cast(UInt64,lr_cast(UInt64,lr_cast(UInt32,uLit1_L36)))) << lr_cast(Int32,32))) | lr_cast(UInt64,(lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,uLit2_L37) << lr_cast(Int32,1))) | lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,fComp))))))))))
    begin _lr_result=lr_cast(UInt64,lr_cast(UInt64,key_L38)); ; return _lr_result end
    
    return nothing
end

# Original cubeHasLit source device body; candidate, not certified.
@inline function lr_cubeHasLit(ctx,uCube,i)::Int32
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,uCube) & lr_cast(UInt32,lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i))))))))) > lr_cast(UInt32,lr_cast(Int32,0)))))); ; return _lr_result end
    
    return nothing
end

# Original cubeSetLit source device body; candidate, not certified.
@inline function lr_cubeSetLit(ctx,uCube,i)::UInt32
    begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,uCube) | lr_cast(UInt32,lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i))))))))); ; return _lr_result end
    
    return nothing
end

# Original cubeRemLit source device body; candidate, not certified.
@inline function lr_cubeRemLit(ctx,uCube,i)::UInt32
    begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,uCube) & lr_cast(UInt32,(~lr_cast(UInt32,lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << lr_cast(Int32,i))))))))))); ; return _lr_result end
    
    return nothing
end

# Original cubeContains source device body; candidate, not certified.
@inline function lr_cubeContains(ctx,uLarge,uSmall)::Int32
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,uLarge) & lr_cast(UInt32,uSmall)))) == lr_cast(UInt32,uSmall))))); ; return _lr_result end
    
    return nothing
end

# Original cubeSharp source device body; candidate, not certified.
@inline function lr_cubeSharp(ctx,uCube,uMask)::UInt32
    begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,uCube) & lr_cast(UInt32,(~lr_cast(UInt32,uMask)))))); ; return _lr_result end
    
    return nothing
end

# Original cubeIsMarked source device body; candidate, not certified.
@inline function lr_cubeIsMarked(ctx,uCube)::Int32
    begin _lr_result=lr_cast(Int32,lr_cubeHasLit(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,31))); ; return _lr_result end
    
    return nothing
end

# Original cubeMark source device body; candidate, not certified.
@inline function lr_cubeMark(ctx,uCube)::UInt32
    begin _lr_result=lr_cast(UInt32,lr_cubeSetLit(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,31))); ; return _lr_result end
    
    return nothing
end

# Original cubeUnmark source device body; candidate, not certified.
@inline function lr_cubeUnmark(ctx,uCube)::UInt32
    begin _lr_result=lr_cast(UInt32,lr_cubeRemLit(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,31))); ; return _lr_result end
    
    return nothing
end

# Original minatoIsop5Rec source device body; candidate, not certified.
@inline function lr_minatoIsop5Rec(ctx,uOn,uOnDc,nVars,pcRes,vecsMem)::UInt32
    _lr_mem_0=lr_alloc(LRArray{UInt32,5})
    uMasks_L13=lr_pointer_cast(UInt32,_lr_mem_0)
    _lr_mem_1=lr_alloc(LRSop)
    cRes0_L14=_lr_mem_1[0]
    _lr_mem_2=lr_alloc(LRSop)
    cRes1_L14=_lr_mem_2[0]
    _lr_mem_3=lr_alloc(LRSop)
    cRes2_L14=_lr_mem_3[0]
    _lr_mem_4=lr_alloc(UInt32)
    uOn0_L16=_lr_mem_4
    _lr_mem_5=lr_alloc(UInt32)
    uOn1_L16=_lr_mem_5
    _lr_mem_6=lr_alloc(UInt32)
    uOnDc0_L16=_lr_mem_6
    _lr_mem_7=lr_alloc(UInt32)
    uOnDc1_L16=_lr_mem_7
    LRPtr{UInt32}(getfield(uMasks_L13,:base))[0]=lr_cast(UInt32,2863311530)
    LRPtr{UInt32}(getfield(uMasks_L13,:base))[1]=lr_cast(UInt32,3435973836)
    LRPtr{UInt32}(getfield(uMasks_L13,:base))[2]=lr_cast(UInt32,4042322160)
    LRPtr{UInt32}(getfield(uMasks_L13,:base))[3]=lr_cast(UInt32,4278255360)
    LRPtr{UInt32}(getfield(uMasks_L13,:base))[4]=lr_cast(UInt32,4294901760)



    pcRes0_L15=lr_pointer_cast(LRSop,cRes0_L14)
    pcRes1_L15=lr_pointer_cast(LRSop,cRes1_L14)
    pcRes2_L15=lr_pointer_cast(LRSop,cRes2_L14)




    local uRes0_L16
    local uRes1_L16
    local uRes2_L16
    local i_L17
    local k_L17
    local Var_L17
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,nVars) <= lr_cast(Int32,5))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,((lr_cast(UInt32,(lr_cast(UInt32,uOn[0]) & lr_cast(UInt32,(~lr_cast(UInt32,uOnDc[0])))))) == lr_cast(UInt32,lr_cast(Int32,0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,uOn[0]) == lr_cast(UInt32,lr_cast(Int32,0)))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,pcRes)).nLits = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,pcRes)).nCubes = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,Int64(0))); (lr_pointer_cast(LRSop,pcRes)).pCubes = _lr_rhs; _lr_rhs)
        begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_5,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_6,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_7,:base))); return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,uOnDc[0]) == lr_cast(UInt32,4294967295))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,pcRes)).nLits = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(LRSop,pcRes)).nCubes = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,1))); (lr_pointer_cast(LRSop,pcRes)).pCubes = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes) != lr_pointer_cast(UInt32,Int64(0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
        begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,4294967295)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_5,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_6,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_7,:base))); return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,nVars) - lr_cast(Int32,1)))); Var_L17 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,Var_L17) >= lr_cast(Int32,0))))
        if lr_truth((lr_truth(lr_cast(Bool,lr_varInSupport(ctx ,lr_pointer_cast(UInt32,uOn),lr_cast(Int32,5),lr_cast(Int32,Var_L17)))) || lr_truth(lr_cast(Bool,lr_varInSupport(ctx ,lr_pointer_cast(UInt32,uOnDc),lr_cast(Int32,5),lr_cast(Int32,Var_L17))))))
            break
        end
        (_lr_old=Var_L17; Var_L17=lr_cast(Int32,(Var_L17-Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,Var_L17) >= lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,uOn[0])); uOn1_L16[0] = _lr_rhs; _lr_rhs))); uOn0_L16[0] = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,uOnDc[0])); uOnDc1_L16[0] = _lr_rhs; _lr_rhs))); uOnDc0_L16[0] = _lr_rhs; _lr_rhs)
    lr_cofactor0(ctx ,uOn0_L16,lr_cast(Int32,(lr_cast(Int32,Var_L17) + lr_cast(Int32,1))),lr_cast(Int32,Var_L17))
    lr_cofactor1(ctx ,uOn1_L16,lr_cast(Int32,(lr_cast(Int32,Var_L17) + lr_cast(Int32,1))),lr_cast(Int32,Var_L17))
    lr_cofactor0(ctx ,uOnDc0_L16,lr_cast(Int32,(lr_cast(Int32,Var_L17) + lr_cast(Int32,1))),lr_cast(Int32,Var_L17))
    lr_cofactor1(ctx ,uOnDc1_L16,lr_cast(Int32,(lr_cast(Int32,Var_L17) + lr_cast(Int32,1))),lr_cast(Int32,Var_L17))
    (_lr_rhs=lr_cast(UInt32,lr_minatoIsop5Rec(ctx ,lr_cast(UInt32,(lr_cast(UInt32,uOn0_L16[0]) & lr_cast(UInt32,(~lr_cast(UInt32,uOnDc1_L16[0]))))),lr_cast(UInt32,uOnDc0_L16[0]),lr_cast(Int32,Var_L17),lr_pointer_cast(LRSop,pcRes0_L15),lr_pointer_cast(LRVecsMem,vecsMem))); uRes0_L16 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_minatoIsop5Rec(ctx ,lr_cast(UInt32,(lr_cast(UInt32,uOn1_L16[0]) & lr_cast(UInt32,(~lr_cast(UInt32,uOnDc0_L16[0]))))),lr_cast(UInt32,uOnDc1_L16[0]),lr_cast(Int32,Var_L17),lr_pointer_cast(LRSop,pcRes1_L15),lr_pointer_cast(LRVecsMem,vecsMem))); uRes1_L16 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_minatoIsop5Rec(ctx ,lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uOn0_L16[0]) & lr_cast(UInt32,(~lr_cast(UInt32,uRes0_L16)))))) | (lr_cast(UInt32,(lr_cast(UInt32,uOn1_L16[0]) & lr_cast(UInt32,(~lr_cast(UInt32,uRes1_L16)))))))),lr_cast(UInt32,(lr_cast(UInt32,uOnDc0_L16[0]) & lr_cast(UInt32,uOnDc1_L16[0]))),lr_cast(Int32,Var_L17),lr_pointer_cast(LRSop,pcRes2_L15),lr_pointer_cast(LRVecsMem,vecsMem))); uRes2_L16 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L15)).nLits) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L15)).nLits))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes2_L15)).nLits))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L15)).nCubes))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L15)).nCubes)))); (lr_pointer_cast(LRSop,pcRes)).nLits = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L15)).nCubes) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L15)).nCubes))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes2_L15)).nCubes)))); (lr_pointer_cast(LRSop,pcRes)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes)).nCubes))); (lr_pointer_cast(LRSop,pcRes)).pCubes = _lr_rhs; _lr_rhs)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes) != lr_pointer_cast(UInt32,Int64(0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L17 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L17 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L17) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L15)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes0_L15)).pCubes))[lr_cast(Int32,i_L17)]) | lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,Var_L17) << lr_cast(Int32,1)))) + lr_cast(Int32,0))))))))))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[(_lr_old=k_L17; k_L17=lr_cast(Int32,(k_L17+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L17; i_L17=lr_cast(Int32,(i_L17+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L17 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L17) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L15)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes1_L15)).pCubes))[lr_cast(Int32,i_L17)]) | lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,Var_L17) << lr_cast(Int32,1)))) + lr_cast(Int32,1))))))))))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[(_lr_old=k_L17; k_L17=lr_cast(Int32,(k_L17+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L17; i_L17=lr_cast(Int32,(i_L17+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L17 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L17) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes2_L15)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes2_L15)).pCubes))[lr_cast(Int32,i_L17)])); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[(_lr_old=k_L17; k_L17=lr_cast(Int32,(k_L17+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L17; i_L17=lr_cast(Int32,(i_L17+Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,k_L17) == lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes)).nCubes))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(uRes2_L16 | lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uRes0_L16) & lr_cast(UInt32,(~lr_cast(UInt32,(lr_pointer_cast(UInt32,uMasks_L13))[lr_cast(Int32,Var_L17)])))))) | (lr_cast(UInt32,(lr_cast(UInt32,uRes1_L16) & lr_cast(UInt32,(lr_pointer_cast(UInt32,uMasks_L13))[lr_cast(Int32,Var_L17)]))))))))); uRes2_L16 = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,uRes2_L16)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_5,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_6,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_7,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_4,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_5,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_6,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_7,:base)))
    return nothing
end

# Original minatoIsopRec source device body; candidate, not certified.
@inline function lr_minatoIsopRec(ctx,puOn,puOnDc,nVars,pcRes,vecsMem)
    _lr_mem_0=lr_alloc(LRSop)
    cRes0_L83=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRSop)
    cRes1_L83=_lr_mem_1[0]
    _lr_mem_2=lr_alloc(LRSop)
    cRes2_L83=_lr_mem_2[0]



    pcRes0_L84=lr_pointer_cast(LRSop,cRes0_L83)
    pcRes1_L84=lr_pointer_cast(LRSop,cRes1_L83)
    pcRes2_L84=lr_pointer_cast(LRSop,cRes2_L83)
    local puRes0_L85
    local puRes1_L85
    local puRes2_L85
    local puOn0_L86
    local puOn1_L86
    local puOnDc0_L86
    local puOnDc1_L86
    local pTemp_L87
    local pTemp0_L87
    local pTemp1_L87
    local i_L88
    local k_L88
    local Var_L88
    local nWords_L88
    local nWordsAll_L88
    (_lr_rhs=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,nVars))); nWordsAll_L88 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,nWordsAll_L88))); pTemp_L87 = _lr_rhs; _lr_rhs)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_pointer_cast(UInt32,pTemp_L87) != lr_pointer_cast(UInt32,Int64(0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,lr_isConst0(ctx ,lr_pointer_cast(UInt32,puOn),lr_cast(Int32,nVars))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,pcRes)).nLits = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,pcRes)).nCubes = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,Int64(0))); (lr_pointer_cast(LRSop,pcRes)).pCubes = _lr_rhs; _lr_rhs)
        lr_clear(ctx ,lr_pointer_cast(UInt32,pTemp_L87),lr_cast(Int32,nVars))
        begin _lr_result=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp_L87)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    end
    if lr_truth(lr_cast(Bool,lr_isConst1(ctx ,lr_pointer_cast(UInt32,puOnDc),lr_cast(Int32,nVars))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,pcRes)).nLits = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); (lr_pointer_cast(LRSop,pcRes)).nCubes = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,1))); (lr_pointer_cast(LRSop,pcRes)).pCubes = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes) != lr_pointer_cast(UInt32,Int64(0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
        lr_fill(ctx ,lr_pointer_cast(UInt32,pTemp_L87),lr_cast(Int32,nVars))
        begin _lr_result=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp_L87)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,nVars) - lr_cast(Int32,1)))); Var_L88 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,Var_L88) >= lr_cast(Int32,0))))
        if lr_truth((lr_truth(lr_cast(Bool,lr_varInSupport(ctx ,lr_pointer_cast(UInt32,puOn),lr_cast(Int32,nVars),lr_cast(Int32,Var_L88)))) || lr_truth(lr_cast(Bool,lr_varInSupport(ctx ,lr_pointer_cast(UInt32,puOnDc),lr_cast(Int32,nVars),lr_cast(Int32,Var_L88))))))
            break
        end
        (_lr_old=Var_L88; Var_L88=lr_cast(Int32,(Var_L88-Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,Var_L88) >= lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,Var_L88) < lr_cast(Int32,5))))
        uRes_L122=lr_cast(UInt32,lr_minatoIsop5Rec(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,puOn))[lr_cast(Int32,0)]),lr_cast(UInt32,(lr_pointer_cast(UInt32,puOnDc))[lr_cast(Int32,0)]),lr_cast(Int32,(lr_cast(Int32,Var_L88) + lr_cast(Int32,1))),lr_pointer_cast(LRSop,pcRes),lr_pointer_cast(LRVecsMem,vecsMem)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L88 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L88) < lr_cast(Int32,nWordsAll_L88))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,uRes_L122)); (lr_pointer_cast(UInt32,pTemp_L87))[lr_cast(Int32,i_L88)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L88; i_L88=lr_cast(Int32,(i_L88+Int32(1))); _lr_old)
        end
        begin _lr_result=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp_L87)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,Var_L88) >= lr_cast(Int32,5))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_factor_TruthWordNum(lr_cast(Int32,Var_L88))); nWords_L88 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puOn)); puOn0_L86 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,puOn) + lr_cast(Int32,nWords_L88)))); puOn1_L86 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puOnDc)); puOnDc0_L86 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,puOnDc) + lr_cast(Int32,nWords_L88)))); puOnDc1_L86 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp_L87)); pTemp0_L87 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,pTemp_L87) + lr_cast(Int32,nWords_L88)))); pTemp1_L87 = _lr_rhs; _lr_rhs)
    lr_truthSharp(ctx ,lr_pointer_cast(UInt32,pTemp0_L87),lr_pointer_cast(UInt32,puOn0_L86),lr_pointer_cast(UInt32,puOnDc1_L86),lr_cast(Int32,Var_L88))
    (_lr_rhs=lr_pointer_cast(UInt32,lr_minatoIsopRec(ctx ,lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp0_L87)),lr_pointer_cast(UInt32,puOnDc0_L86),lr_cast(Int32,Var_L88),lr_pointer_cast(LRSop,pcRes0_L84),lr_pointer_cast(LRVecsMem,vecsMem))); puRes0_L85 = _lr_rhs; _lr_rhs)
    lr_truthSharp(ctx ,lr_pointer_cast(UInt32,pTemp1_L87),lr_pointer_cast(UInt32,puOn1_L86),lr_pointer_cast(UInt32,puOnDc0_L86),lr_cast(Int32,Var_L88))
    (_lr_rhs=lr_pointer_cast(UInt32,lr_minatoIsopRec(ctx ,lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp1_L87)),lr_pointer_cast(UInt32,puOnDc1_L86),lr_cast(Int32,Var_L88),lr_pointer_cast(LRSop,pcRes1_L84),lr_pointer_cast(LRVecsMem,vecsMem))); puRes1_L85 = _lr_rhs; _lr_rhs)
    lr_truthSharp(ctx ,lr_pointer_cast(UInt32,pTemp0_L87),lr_pointer_cast(UInt32,puOn0_L86),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puRes0_L85)),lr_cast(Int32,Var_L88))
    lr_truthSharp(ctx ,lr_pointer_cast(UInt32,pTemp1_L87),lr_pointer_cast(UInt32,puOn1_L86),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puRes1_L85)),lr_cast(Int32,Var_L88))
    lr_truthOr(ctx ,lr_pointer_cast(UInt32,pTemp0_L87),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp0_L87)),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp1_L87)),lr_cast(Int32,Var_L88))
    lr_truthAnd(ctx ,lr_pointer_cast(UInt32,pTemp1_L87),lr_pointer_cast(UInt32,puOnDc0_L86),lr_pointer_cast(UInt32,puOnDc1_L86),lr_cast(Int32,Var_L88))
    (_lr_rhs=lr_pointer_cast(UInt32,lr_minatoIsopRec(ctx ,lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp0_L87)),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp1_L87)),lr_cast(Int32,Var_L88),lr_pointer_cast(LRSop,pcRes2_L84),lr_pointer_cast(LRVecsMem,vecsMem))); puRes2_L85 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L84)).nLits) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L84)).nLits))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes2_L84)).nLits))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L84)).nCubes))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L84)).nCubes)))); (lr_pointer_cast(LRSop,pcRes)).nLits = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L84)).nCubes) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L84)).nCubes))) + lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes2_L84)).nCubes)))); (lr_pointer_cast(LRSop,pcRes)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes)).nCubes))); (lr_pointer_cast(LRSop,pcRes)).pCubes = _lr_rhs; _lr_rhs)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes) != lr_pointer_cast(UInt32,Int64(0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L88 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L88 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L88) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes0_L84)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes0_L84)).pCubes))[lr_cast(Int32,i_L88)]) | lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,Var_L88) << lr_cast(Int32,1)))) + lr_cast(Int32,0))))))))))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[(_lr_old=k_L88; k_L88=lr_cast(Int32,(k_L88+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L88; i_L88=lr_cast(Int32,(i_L88+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L88 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L88) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes1_L84)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes1_L84)).pCubes))[lr_cast(Int32,i_L88)]) | lr_cast(UInt32,(lr_cast(Int32,(lr_cast(Int32,1) << (lr_cast(Int32,((lr_cast(Int32,(lr_cast(Int32,Var_L88) << lr_cast(Int32,1)))) + lr_cast(Int32,1))))))))))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[(_lr_old=k_L88; k_L88=lr_cast(Int32,(k_L88+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L88; i_L88=lr_cast(Int32,(i_L88+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L88 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L88) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes2_L84)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes2_L84)).pCubes))[lr_cast(Int32,i_L88)])); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes)).pCubes))[(_lr_old=k_L88; k_L88=lr_cast(Int32,(k_L88+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L88; i_L88=lr_cast(Int32,(i_L88+Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,k_L88) == lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes)).nCubes))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    lr_truthOr(ctx ,lr_pointer_cast(UInt32,pTemp0_L87),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puRes0_L85)),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puRes2_L85)),lr_cast(Int32,Var_L88))
    lr_truthOr(ctx ,lr_pointer_cast(UInt32,pTemp1_L87),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puRes1_L85)),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,puRes2_L85)),lr_cast(Int32,Var_L88))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(nWords_L88 << lr_cast(Int32,1)))); nWords_L88 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); i_L88 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L88) < lr_cast(Int32,div(lr_cast(Int32,nWordsAll_L88),lr_cast(Int32,nWords_L88))))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L88 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L88) < lr_cast(Int32,nWords_L88))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,pTemp_L87))[lr_cast(Int32,k_L88)])); (lr_pointer_cast(UInt32,pTemp_L87))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,i_L88) * lr_cast(Int32,nWords_L88))) + lr_cast(Int32,k_L88)))] = _lr_rhs; _lr_rhs)
            (_lr_old=k_L88; k_L88=lr_cast(Int32,(k_L88+Int32(1))); _lr_old)
        end
        (_lr_old=i_L88; i_L88=lr_cast(Int32,(i_L88+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pTemp_L87)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base)))
    return nothing
end

# Original minatoIsop source device body; candidate, not certified.
@inline function lr_minatoIsop(ctx,puTruth,nVars,vecsMem)::Nothing
    _lr_mem_0=lr_alloc(LRSop)
    cRes_L178=_lr_mem_0[0]

    pcRes_L178=lr_pointer_cast(LRSop,cRes_L178)
    local pResult_L179
    local pTemp_L179
    lr_factor_shrink(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,0))
    (_lr_rhs=lr_pointer_cast(UInt32,lr_minatoIsopRec(ctx ,lr_pointer_cast(UInt32,puTruth),lr_pointer_cast(UInt32,puTruth),lr_cast(Int32,nVars),lr_pointer_cast(LRSop,pcRes_L178),lr_pointer_cast(LRVecsMem,vecsMem))); pResult_L179 = _lr_rhs; _lr_rhs)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,lr_truthEqual(ctx ,lr_pointer_cast(UInt32,puTruth),lr_pointer_cast(UInt32,lr_pointer_cast(UInt32,pResult_L179)),lr_cast(Int32,nVars))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes) == lr_cast(Int32,0)))) || lr_truth(((lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes) == lr_cast(Int32,1)))) && lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes_L178)).pCubes))[lr_cast(Int32,0)]) == lr_cast(UInt32,lr_cast(Int32,0))))))))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRVecsMem,vecsMem)).pArray))[lr_cast(Int32,0)] = _lr_rhs; _lr_rhs)
        lr_factor_shrink(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes))
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); return _lr_result end
    end
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes))); pTemp_L179 = _lr_rhs; _lr_rhs)
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_pointer_cast(UInt32,pTemp_L179) != lr_pointer_cast(UInt32,Int64(0)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    i_L195=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L195) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,pcRes_L178)).pCubes))[lr_cast(Int32,i_L195)])); (lr_pointer_cast(UInt32,pTemp_L179))[lr_cast(Int32,i_L195)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L195; i_L195=lr_cast(Int32,(i_L195+Int32(1))); _lr_old)
    end
    i_L197=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L197) < lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,pTemp_L179))[lr_cast(Int32,i_L197)])); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRVecsMem,vecsMem)).pArray))[lr_cast(Int32,i_L197)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L197; i_L197=lr_cast(Int32,(i_L197+Int32(1))); _lr_old)
    end
    lr_factor_shrink(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,pcRes_L178)).nCubes))
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base)))
    return nothing
end

# Original sopCreateInverse source device body; candidate, not certified.
@inline function lr_sopCreateInverse(ctx,cResult,vInput,nInputCubes,vecsMem)::Nothing
    local uCube_L22
    uMask_L22=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,cResult)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,nInputCubes))); (lr_pointer_cast(LRSop,cResult)).pCubes = _lr_rhs; _lr_rhs)
    i_L27=lr_cast(Int32,lr_cast(Int32,0))
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L27) < lr_cast(Int32,nInputCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,vInput))[lr_cast(Int32,i_L27)])); uCube_L22 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,(lr_cast(UInt32,((lr_cast(UInt32,(lr_cast(UInt32,uCube_L22) | (lr_cast(UInt32,(lr_cast(UInt32,uCube_L22) >> lr_cast(Int32,1))))))) & lr_cast(UInt32,lr_cast(Int32,1431655765)))))); uMask_L22 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(uMask_L22 | (lr_cast(UInt32,(lr_cast(UInt32,uMask_L22) << lr_cast(Int32,1))))))); uMask_L22 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,xor(lr_cast(UInt32,uCube_L22),lr_cast(UInt32,uMask_L22)))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cResult)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,cResult)).nCubes; (lr_pointer_cast(LRSop,cResult)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,cResult)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L27; i_L27=lr_cast(Int32,(i_L27+Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original sopCommonCube source device body; candidate, not certified.
@inline function lr_sopCommonCube(ctx,cSop)::UInt32
    local uMask_L46
    local i_L47
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(~lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0)))))); uMask_L46 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L47 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L47) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(uMask_L46 & lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L47)])))); uMask_L46 = _lr_rhs; _lr_rhs)
        (_lr_old=i_L47; i_L47=lr_cast(Int32,(i_L47+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(UInt32,lr_cast(UInt32,uMask_L46)); ; return _lr_result end
    
    return nothing
end

# Original sopMakeCubeFree source device body; candidate, not certified.
@inline function lr_sopMakeCubeFree(ctx,cSop)::Nothing
    local uMask_L56
    local i_L57
    (_lr_rhs=lr_cast(UInt32,lr_sopCommonCube(ctx ,lr_pointer_cast(LRSop,cSop))); uMask_L56 = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_cast(UInt32,uMask_L56) == lr_cast(UInt32,lr_cast(Int32,0)))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L57 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L57) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cubeSharp(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L57)]),lr_cast(UInt32,uMask_L56))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L57)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L57; i_L57=lr_cast(Int32,(i_L57+Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original sopDivideByLiteralQuo source device body; candidate, not certified.
@inline function lr_sopDivideByLiteralQuo(ctx,cSop,iLit)::Nothing
    local i_L69
    k_L69=lr_cast(Int32,lr_cast(Int32,0))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L69 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L69) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        if lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L69)]),lr_cast(Int32,iLit))))
            (_lr_rhs=lr_cast(UInt32,lr_cubeRemLit(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L69)]),lr_cast(Int32,iLit))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[(_lr_old=k_L69; k_L69=lr_cast(Int32,(k_L69+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        end
        (_lr_old=i_L69; i_L69=lr_cast(Int32,(i_L69+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,k_L69)); (lr_pointer_cast(LRSop,cSop)).nCubes = _lr_rhs; _lr_rhs)
    
    return nothing
end

# Original sopWorstLiteral source device body; candidate, not certified.
@inline function lr_sopWorstLiteral(ctx,cSop,nLits)::Int32
    local i_L78
    local k_L78
    local iMin_L78
    local nLitsMin_L78
    local nLitsCur_L78
    fUseFirst_L79=lr_cast(Int32,lr_cast(Int32,1))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); iMin_L78 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1000000)); nLitsMin_L78 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L78 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L78) < lr_cast(Int32,nLits))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); nLitsCur_L78 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L78 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L78) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
            if lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,k_L78)]),lr_cast(Int32,i_L78))))
                (_lr_old=nLitsCur_L78; nLitsCur_L78=lr_cast(Int32,(nLitsCur_L78+Int32(1))); _lr_old)
            end
            (_lr_old=k_L78; k_L78=lr_cast(Int32,(k_L78+Int32(1))); _lr_old)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsCur_L78) < lr_cast(Int32,2))))
            (_lr_old=i_L78; i_L78=lr_cast(Int32,(i_L78+Int32(1))); _lr_old); continue
        end
        if lr_truth(lr_cast(Bool,lr_cast(Int32,fUseFirst_L79)))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsMin_L78) > lr_cast(Int32,nLitsCur_L78))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nLitsCur_L78)); nLitsMin_L78 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L78)); iMin_L78 = _lr_rhs; _lr_rhs)
            end
        else
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsMin_L78) >= lr_cast(Int32,nLitsCur_L78))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nLitsCur_L78)); nLitsMin_L78 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L78)); iMin_L78 = _lr_rhs; _lr_rhs)
            end
        end
        (_lr_old=i_L78; i_L78=lr_cast(Int32,(i_L78+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsMin_L78) < lr_cast(Int32,1000000))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,iMin_L78)); ; return _lr_result end
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); ; return _lr_result end
    
    return nothing
end

# Original sopDivisorZeroKernelRec source device body; candidate, not certified.
@inline function lr_sopDivisorZeroKernelRec(ctx,cSop,nLits)::Nothing
    local iLit_L114
    (_lr_rhs=lr_cast(Int32,lr_sopWorstLiteral(ctx ,lr_pointer_cast(LRSop,cSop),lr_cast(Int32,nLits))); iLit_L114 = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,iLit_L114) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    lr_sopDivideByLiteralQuo(ctx ,lr_pointer_cast(LRSop,cSop),lr_cast(Int32,iLit_L114))
    lr_sopMakeCubeFree(ctx ,lr_pointer_cast(LRSop,cSop))
    lr_sopDivisorZeroKernelRec(ctx ,lr_pointer_cast(LRSop,cSop),lr_cast(Int32,nLits))
    
    return nothing
end

# Original sopDup source device body; candidate, not certified.
@inline function lr_sopDup(ctx,cResult,cSop,vecsMem)::Nothing
    local i_L128
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,cResult)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))); (lr_pointer_cast(LRSop,cResult)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L128 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L128) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L128)])); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cResult)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,cResult)).nCubes; (lr_pointer_cast(LRSop,cResult)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,cResult)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L128; i_L128=lr_cast(Int32,(i_L128+Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original sopAnyLiteral source device body; candidate, not certified.
@inline function lr_sopAnyLiteral(ctx,cSop,nLits)::Int32
    local i_L140
    local k_L140
    local nLitsCur_L140
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L140 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L140) < lr_cast(Int32,nLits))))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); nLitsCur_L140 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L140 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L140) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
            if lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,k_L140)]),lr_cast(Int32,i_L140))))
                (_lr_old=nLitsCur_L140; nLitsCur_L140=lr_cast(Int32,(nLitsCur_L140+Int32(1))); _lr_old)
            end
            (_lr_old=k_L140; k_L140=lr_cast(Int32,(k_L140+Int32(1))); _lr_old)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsCur_L140) > lr_cast(Int32,1))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,i_L140)); ; return _lr_result end
        end
        (_lr_old=i_L140; i_L140=lr_cast(Int32,(i_L140+Int32(1))); _lr_old)
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); ; return _lr_result end
    
    return nothing
end

# Original sopDivisor source device body; candidate, not certified.
@inline function lr_sopDivisor(ctx,cResult,cSop,nLits,vecsMem)::Int32
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes) <= lr_cast(Int32,1))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    if lr_truth(lr_cast(Bool,(lr_sopAnyLiteral(ctx ,lr_pointer_cast(LRSop,cSop),lr_cast(Int32,nLits)) == lr_cast(Int32,(-lr_cast(Int32,1))))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,0)); ; return _lr_result end
    end
    lr_sopDup(ctx ,lr_pointer_cast(LRSop,cResult),lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRVecsMem,vecsMem))
    lr_sopDivisorZeroKernelRec(ctx ,lr_pointer_cast(LRSop,cResult),lr_cast(Int32,nLits))
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cResult)).nCubes) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,1)); ; return _lr_result end
    
    return nothing
end

# Original sopFactorTrivialCubeRec source device body; candidate, not certified.
@inline function lr_sopFactorTrivialCubeRec(ctx,uCube,nStart,nFinish,subg)::Int32
    local eNode1_L173
    local eNode2_L173
    local i_L174
    iLit_L174=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1))))
    local nLits_L174
    local nLits1_L174
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,lr_cast(UInt32,uCube)))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); nLits_L174 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nStart)); i_L174 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L174) < lr_cast(Int32,nFinish))))
        if lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,i_L174))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L174)); iLit_L174 = _lr_rhs; _lr_rhs)
            (_lr_old=nLits_L174; nLits_L174=lr_cast(Int32,(nLits_L174+Int32(1))); _lr_old)
        end
        (_lr_old=i_L174; i_L174=lr_cast(Int32,(i_L174+Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,iLit_L174) != lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLits_L174) == lr_cast(Int32,1))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,iLit_L174)); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,div(lr_cast(Int32,nLits_L174),lr_cast(Int32,2)))); nLits1_L174 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); nLits_L174 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nStart)); i_L174 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L174) < lr_cast(Int32,nFinish))))
        if lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,i_L174))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLits_L174) == lr_cast(Int32,nLits1_L174))))
                break
            end
            (_lr_old=nLits_L174; nLits_L174=lr_cast(Int32,(nLits_L174+Int32(1))); _lr_old)
        end
        (_lr_old=i_L174; i_L174=lr_cast(Int32,(i_L174+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_sopFactorTrivialCubeRec(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,nStart),lr_cast(Int32,i_L174),lr_pointer_cast(LRSubg,subg))); eNode1_L173 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_sopFactorTrivialCubeRec(ctx ,lr_cast(UInt32,uCube),lr_cast(Int32,i_L174),lr_cast(Int32,nFinish),lr_pointer_cast(LRSubg,subg))); eNode2_L173 = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_factor_addNodeAnd(lr_pointer_cast(LRSubg,subg) ,lr_cast(Int32,eNode1_L173),lr_cast(Int32,eNode2_L173))); ; return _lr_result end
    
    return nothing
end

# Original sopFactorTrivialRec source device body; candidate, not certified.
@inline function lr_sopFactorTrivialRec(ctx,pCubes,nCubes,nLits,subg)::Int32
    local eNode1_L210
    local eNode2_L210
    local nCubes1_L211
    local nCubes2_L211
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nCubes) == lr_cast(Int32,1))))
        begin _lr_result=lr_cast(Int32,lr_sopFactorTrivialCubeRec(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,pCubes))[lr_cast(Int32,0)]),lr_cast(Int32,0),lr_cast(Int32,nLits),lr_pointer_cast(LRSubg,subg))); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,div(lr_cast(Int32,nCubes),lr_cast(Int32,2)))); nCubes1_L211 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,nCubes) - lr_cast(Int32,nCubes1_L211)))); nCubes2_L211 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_sopFactorTrivialRec(ctx ,lr_pointer_cast(UInt32,pCubes),lr_cast(Int32,nCubes1_L211),lr_cast(Int32,nLits),lr_pointer_cast(LRSubg,subg))); eNode1_L210 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_sopFactorTrivialRec(ctx ,lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,pCubes) + lr_cast(Int32,nCubes1_L211))),lr_cast(Int32,nCubes2_L211),lr_cast(Int32,nLits),lr_pointer_cast(LRSubg,subg))); eNode2_L210 = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_factor_addNodeOr(lr_pointer_cast(LRSubg,subg) ,lr_cast(Int32,eNode1_L210),lr_cast(Int32,eNode2_L210))); ; return _lr_result end
    
    return nothing
end

# Original sopDivideByCube source device body; candidate, not certified.
@inline function lr_sopDivideByCube(ctx,cSop,cDiv,vQuo,vRem,vecsMem)::Nothing
    local uCube_L228
    local uDiv_L228
    local i_L229
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes) == lr_cast(Int32,1))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cDiv)).pCubes))[lr_cast(Int32,0)])); uDiv_L228 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,vQuo)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))); (lr_pointer_cast(LRSop,vQuo)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,vRem)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))); (lr_pointer_cast(LRSop,vRem)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L229 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L229) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L229)])); uCube_L228 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,lr_cubeContains(ctx ,lr_cast(UInt32,uCube_L228),lr_cast(UInt32,uDiv_L228))))
            (_lr_rhs=lr_cast(UInt32,lr_cubeSharp(ctx ,lr_cast(UInt32,uCube_L228),lr_cast(UInt32,uDiv_L228))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,vQuo)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,vQuo)).nCubes; (lr_pointer_cast(LRSop,vQuo)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,vQuo)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        else
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,uCube_L228)); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,vRem)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,vRem)).nCubes; (lr_pointer_cast(LRSop,vRem)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,vRem)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        end
        (_lr_old=i_L229; i_L229=lr_cast(Int32,(i_L229+Int32(1))); _lr_old)
    end
    
    return nothing
end

# Original sopDivideInternal source device body; candidate, not certified.
@inline function lr_sopDivideInternal(ctx,cSop,cDiv,vQuo,vRem,vecsMem)::Nothing
    local uCube_L253
    local uDiv_L253
    uCube2_L254=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,0)))
    local uDiv2_L255
    local uQuo_L255
    local i_L256
    local i2_L256
    local k_L256
    local k2_L256
    local nCubesRem_L256
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes) >= lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes) == lr_cast(Int32,1))))
        lr_sopDivideByCube(ctx ,lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRSop,cDiv),lr_pointer_cast(LRSop,vQuo),lr_pointer_cast(LRSop,vRem),lr_pointer_cast(LRVecsMem,vecsMem))
        begin _lr_result=(nothing; nothing); ; return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,vQuo)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,div(lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes),lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))); (lr_pointer_cast(LRSop,vQuo)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L256 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L256)])); uCube_L253 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,lr_cubeIsMarked(ctx ,lr_cast(UInt32,uCube_L253))))
            (_lr_old=i_L256; i_L256=lr_cast(Int32,(i_L256+Int32(1))); _lr_old); continue
        end
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(~lr_cast(Int32,0))))); uDiv_L253 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L256 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cDiv)).pCubes))[lr_cast(Int32,k_L256)])); uDiv_L253 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,lr_cubeContains(ctx ,lr_cast(UInt32,uCube_L253),lr_cast(UInt32,uDiv_L253))))
                break
            end
            (_lr_old=k_L256; k_L256=lr_cast(Int32,(k_L256+Int32(1))); _lr_old)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L256) == lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))
            (_lr_old=i_L256; i_L256=lr_cast(Int32,(i_L256+Int32(1))); _lr_old); continue
        end
        (_lr_rhs=lr_cast(UInt32,lr_cubeSharp(ctx ,lr_cast(UInt32,uCube_L253),lr_cast(UInt32,uDiv_L253))); uQuo_L255 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,lr_cast(Int32,(~lr_cast(Int32,0))))); uDiv2_L255 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k2_L256 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k2_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cDiv)).pCubes))[lr_cast(Int32,k2_L256)])); uDiv2_L255 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,k2_L256) == lr_cast(Int32,k_L256))))
                (_lr_old=k2_L256; k2_L256=lr_cast(Int32,(k2_L256+Int32(1))); _lr_old); continue
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i2_L256 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i2_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i2_L256)])); uCube2_L254 = _lr_rhs; _lr_rhs)
                if lr_truth(lr_cast(Bool,lr_cubeIsMarked(ctx ,lr_cast(UInt32,uCube2_L254))))
                    (_lr_old=i2_L256; i2_L256=lr_cast(Int32,(i2_L256+Int32(1))); _lr_old); continue
                end
                if lr_truth((lr_truth(lr_cast(Bool,lr_cubeContains(ctx ,lr_cast(UInt32,uCube2_L254),lr_cast(UInt32,uDiv2_L255)))) && lr_truth(lr_cast(Bool,(lr_cast(UInt32,uQuo_L255) == lr_cubeSharp(ctx ,lr_cast(UInt32,uCube2_L254),lr_cast(UInt32,uDiv2_L255)))))))
                    break
                end
                (_lr_old=i2_L256; i2_L256=lr_cast(Int32,(i2_L256+Int32(1))); _lr_old)
            end
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,i2_L256) == lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
                break
            end
            (_lr_old=k2_L256; k2_L256=lr_cast(Int32,(k2_L256+Int32(1))); _lr_old)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,k2_L256) != lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))
            (_lr_old=i_L256; i_L256=lr_cast(Int32,(i_L256+Int32(1))); _lr_old); continue
        end
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,uQuo_L255)); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,vQuo)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,vQuo)).nCubes; (lr_pointer_cast(LRSop,vQuo)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,vQuo)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(UInt32,lr_cubeMark(ctx ,lr_cast(UInt32,uCube_L253))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L256)] = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k2_L256 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k2_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cDiv)).pCubes))[lr_cast(Int32,k2_L256)])); uDiv2_L255 = _lr_rhs; _lr_rhs)
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,k2_L256) == lr_cast(Int32,k_L256))))
                (_lr_old=k2_L256; k2_L256=lr_cast(Int32,(k2_L256+Int32(1))); _lr_old); continue
            end
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i2_L256 = _lr_rhs; _lr_rhs)
            while lr_truth(lr_cast(Bool,(lr_cast(Int32,i2_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
                (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i2_L256)])); uCube2_L254 = _lr_rhs; _lr_rhs)
                if lr_truth(lr_cast(Bool,lr_cubeIsMarked(ctx ,lr_cast(UInt32,uCube2_L254))))
                    (_lr_old=i2_L256; i2_L256=lr_cast(Int32,(i2_L256+Int32(1))); _lr_old); continue
                end
                if lr_truth((lr_truth(lr_cast(Bool,lr_cubeContains(ctx ,lr_cast(UInt32,uCube2_L254),lr_cast(UInt32,uDiv2_L255)))) && lr_truth(lr_cast(Bool,(lr_cast(UInt32,uQuo_L255) == lr_cubeSharp(ctx ,lr_cast(UInt32,uCube2_L254),lr_cast(UInt32,uDiv2_L255)))))))
                    break
                end
                (_lr_old=i2_L256; i2_L256=lr_cast(Int32,(i2_L256+Int32(1))); _lr_old)
            end
            ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,i2_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
            (_lr_rhs=lr_cast(UInt32,lr_cubeMark(ctx ,lr_cast(UInt32,uCube2_L254))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i2_L256)] = _lr_rhs; _lr_rhs)
            (_lr_old=k2_L256; k2_L256=lr_cast(Int32,(k2_L256+Int32(1))); _lr_old)
        end
        (_lr_old=i_L256; i_L256=lr_cast(Int32,(i_L256+Int32(1))); _lr_old)
    end
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes) - lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSop,vQuo)).nCubes) * lr_cast(Int32,(lr_pointer_cast(LRSop,cDiv)).nCubes)))))); nCubesRem_L256 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,vRem)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,nCubesRem_L256))); (lr_pointer_cast(LRSop,vRem)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L256 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L256) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
        (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L256)])); uCube_L253 = _lr_rhs; _lr_rhs)
        if lr_truth((!lr_truth(lr_cast(Bool,lr_cubeIsMarked(ctx ,lr_cast(UInt32,uCube_L253))))))
            (_lr_rhs=lr_cast(UInt32,lr_cast(UInt32,uCube_L253)); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,vRem)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,vRem)).nCubes; (lr_pointer_cast(LRSop,vRem)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,vRem)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L256; i_L256=lr_cast(Int32,(i_L256+Int32(1))); _lr_old); continue
        end
        (_lr_rhs=lr_cast(UInt32,lr_cubeUnmark(ctx ,lr_cast(UInt32,uCube_L253))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,i_L256)] = _lr_rhs; _lr_rhs)
        (_lr_old=i_L256; i_L256=lr_cast(Int32,(i_L256+Int32(1))); _lr_old)
    end
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,nCubesRem_L256) == lr_cast(Int32,(lr_pointer_cast(LRSop,vRem)).nCubes))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    
    return nothing
end

# Original sopBestLiteral source device body; candidate, not certified.
@inline function lr_sopBestLiteral(ctx,cSop,nLits,uMask)::Int32
    local i_L363
    local k_L363
    local iMax_L363
    local nLitsMax_L363
    local nLitsCur_L363
    fUseFirst_L364=lr_cast(Int32,lr_cast(Int32,1))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); iMax_L363 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); nLitsMax_L363 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); i_L363 = _lr_rhs; _lr_rhs)
    while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L363) < lr_cast(Int32,nLits))))
        if lr_truth((!lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,uMask),lr_cast(Int32,i_L363))))))
            (_lr_old=i_L363; i_L363=lr_cast(Int32,(i_L363+Int32(1))); _lr_old); continue
        end
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); nLitsCur_L363 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); k_L363 = _lr_rhs; _lr_rhs)
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,k_L363) < lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes))))
            if lr_truth(lr_cast(Bool,lr_cubeHasLit(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes))[lr_cast(Int32,k_L363)]),lr_cast(Int32,i_L363))))
                (_lr_old=nLitsCur_L363; nLitsCur_L363=lr_cast(Int32,(nLitsCur_L363+Int32(1))); _lr_old)
            end
            (_lr_old=k_L363; k_L363=lr_cast(Int32,(k_L363+Int32(1))); _lr_old)
        end
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsCur_L363) < lr_cast(Int32,2))))
            (_lr_old=i_L363; i_L363=lr_cast(Int32,(i_L363+Int32(1))); _lr_old); continue
        end
        if lr_truth(lr_cast(Bool,lr_cast(Int32,fUseFirst_L364)))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsMax_L363) < lr_cast(Int32,nLitsCur_L363))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nLitsCur_L363)); nLitsMax_L363 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L363)); iMax_L363 = _lr_rhs; _lr_rhs)
            end
        else
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsMax_L363) <= lr_cast(Int32,nLitsCur_L363))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nLitsCur_L363)); nLitsMax_L363 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,i_L363)); iMax_L363 = _lr_rhs; _lr_rhs)
            end
        end
        (_lr_old=i_L363; i_L363=lr_cast(Int32,(i_L363+Int32(1))); _lr_old)
    end
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nLitsMax_L363) >= lr_cast(Int32,0))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,iMax_L363)); ; return _lr_result end
    end
    begin _lr_result=lr_cast(Int32,lr_cast(Int32,(-lr_cast(Int32,1)))); ; return _lr_result end
    
    return nothing
end

# Original sopBestLiteralCover source device body; candidate, not certified.
@inline function lr_sopBestLiteralCover(ctx,cResult,cSop,uCube,nLits,vecsMem)::Nothing
    local iLitBest_L402
    (_lr_rhs=lr_cast(Int32,lr_sopBestLiteral(ctx ,lr_pointer_cast(LRSop,cSop),lr_cast(Int32,nLits),lr_cast(UInt32,uCube))); iLitBest_L402 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,cResult)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,1))); (lr_pointer_cast(LRSop,cResult)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_cubeSetLit(ctx ,lr_cast(UInt32,lr_cast(Int32,0)),lr_cast(Int32,iLitBest_L402))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cResult)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,cResult)).nCubes; (lr_pointer_cast(LRSop,cResult)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,cResult)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    
    return nothing
end

# Original sopCommonCubeCover source device body; candidate, not certified.
@inline function lr_sopCommonCubeCover(ctx,cResult,cSop,vecsMem)::Nothing
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(LRSop,cResult)).nCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_pointer_cast(UInt32,lr_factor_fetch(lr_pointer_cast(LRVecsMem,vecsMem) ,lr_cast(Int32,1))); (lr_pointer_cast(LRSop,cResult)).pCubes = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(UInt32,lr_sopCommonCube(ctx ,lr_pointer_cast(LRSop,cSop))); (lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cResult)).pCubes))[(_lr_old=(lr_pointer_cast(LRSop,cResult)).nCubes; (lr_pointer_cast(LRSop,cResult)).nCubes=lr_cast(Int32,((lr_pointer_cast(LRSop,cResult)).nCubes+Int32(1))); _lr_old)] = _lr_rhs; _lr_rhs)
    
    return nothing
end

# Original sopFactorRec source device body; candidate, not certified.
@inline function lr_sopFactorRec(ctx,cSop,nLits,vecsMem,subg)::Int32
    _lr_mem_0=lr_alloc(LRSop)
    Div_L425=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRSop)
    Quo_L425=_lr_mem_1[0]
    _lr_mem_2=lr_alloc(LRSop)
    Rem_L425=_lr_mem_2[0]
    _lr_mem_3=lr_alloc(LRSop)
    Com_L425=_lr_mem_3[0]




    cDiv_L426=lr_pointer_cast(LRSop,Div_L425)
    cQuo_L426=lr_pointer_cast(LRSop,Quo_L425)
    cRem_L426=lr_pointer_cast(LRSop,Rem_L425)
    cCom_L426=lr_pointer_cast(LRSop,Com_L425)
    local eNodeDiv_L427
    local eNodeQuo_L427
    local eNodeRem_L427
    local eNodeAnd_L427
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth((!lr_truth(lr_cast(Bool,lr_sopDivisor(ctx ,lr_pointer_cast(LRSop,cDiv_L426),lr_pointer_cast(LRSop,cSop),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem))))))
        begin _lr_result=lr_cast(Int32,lr_sopFactorTrivialRec(ctx ,lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSop)).pCubes),lr_cast(Int32,(lr_pointer_cast(LRSop,cSop)).nCubes),lr_cast(Int32,nLits),lr_pointer_cast(LRSubg,subg))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    end
    lr_sopDivideInternal(ctx ,lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRSop,cDiv_L426),lr_pointer_cast(LRSop,cQuo_L426),lr_pointer_cast(LRSop,cRem_L426),lr_pointer_cast(LRVecsMem,vecsMem))
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cQuo_L426)).nCubes) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cQuo_L426)).nCubes) == lr_cast(Int32,1))))
        begin _lr_result=lr_cast(Int32,lr_sopFactorLFRec(ctx ,lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRSop,cQuo_L426),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    end
    lr_sopMakeCubeFree(ctx ,lr_pointer_cast(LRSop,cQuo_L426))
    lr_sopDivideInternal(ctx ,lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRSop,cQuo_L426),lr_pointer_cast(LRSop,cDiv_L426),lr_pointer_cast(LRSop,cRem_L426),lr_pointer_cast(LRVecsMem,vecsMem))
    if lr_truth(lr_cast(Bool,(lr_sopCommonCube(ctx ,lr_pointer_cast(LRSop,cDiv_L426)) == lr_cast(UInt32,lr_cast(Int32,0)))))
        (_lr_rhs=lr_cast(Int32,lr_sopFactorRec(ctx ,lr_pointer_cast(LRSop,cDiv_L426),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); eNodeDiv_L427 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_sopFactorRec(ctx ,lr_pointer_cast(LRSop,cQuo_L426),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); eNodeQuo_L427 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_factor_addNodeAnd(lr_pointer_cast(LRSubg,subg) ,lr_cast(Int32,eNodeDiv_L427),lr_cast(Int32,eNodeQuo_L427))); eNodeAnd_L427 = _lr_rhs; _lr_rhs)
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cRem_L426)).nCubes) == lr_cast(Int32,0))))
            begin _lr_result=lr_cast(Int32,lr_cast(Int32,eNodeAnd_L427)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
        end
        (_lr_rhs=lr_cast(Int32,lr_sopFactorRec(ctx ,lr_pointer_cast(LRSop,cRem_L426),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); eNodeRem_L427 = _lr_rhs; _lr_rhs)
        begin _lr_result=lr_cast(Int32,lr_factor_addNodeOr(lr_pointer_cast(LRSubg,subg) ,lr_cast(Int32,eNodeAnd_L427),lr_cast(Int32,eNodeRem_L427))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    end
    lr_sopCommonCubeCover(ctx ,lr_pointer_cast(LRSop,cCom_L426),lr_pointer_cast(LRSop,cDiv_L426),lr_pointer_cast(LRVecsMem,vecsMem))
    begin _lr_result=lr_cast(Int32,lr_sopFactorLFRec(ctx ,lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRSop,cCom_L426),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base)))
    return nothing
end

# Original sopFactorLFRec source device body; candidate, not certified.
@inline function lr_sopFactorLFRec(ctx,cSop,cSimple,nLits,vecsMem,subg)::Int32
    _lr_mem_0=lr_alloc(LRSop)
    Div_L471=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRSop)
    Quo_L471=_lr_mem_1[0]
    _lr_mem_2=lr_alloc(LRSop)
    Rem_L471=_lr_mem_2[0]



    cDiv_L472=lr_pointer_cast(LRSop,Div_L471)
    cQuo_L472=lr_pointer_cast(LRSop,Quo_L471)
    cRem_L472=lr_pointer_cast(LRSop,Rem_L471)
    local eNodeDiv_L473
    local eNodeQuo_L473
    local eNodeRem_L473
    local eNodeAnd_L473
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cSimple)).nCubes) == lr_cast(Int32,1))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    lr_sopBestLiteralCover(ctx ,lr_pointer_cast(LRSop,cDiv_L472),lr_pointer_cast(LRSop,cSop),lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cSimple)).pCubes))[lr_cast(Int32,0)]),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem))
    lr_sopDivideByCube(ctx ,lr_pointer_cast(LRSop,cSop),lr_pointer_cast(LRSop,cDiv_L472),lr_pointer_cast(LRSop,cQuo_L472),lr_pointer_cast(LRSop,cRem_L472),lr_pointer_cast(LRVecsMem,vecsMem))
    (_lr_rhs=lr_cast(Int32,lr_sopFactorTrivialCubeRec(ctx ,lr_cast(UInt32,(lr_pointer_cast(UInt32,(lr_pointer_cast(LRSop,cDiv_L472)).pCubes))[lr_cast(Int32,0)]),lr_cast(Int32,0),lr_cast(Int32,nLits),lr_pointer_cast(LRSubg,subg))); eNodeDiv_L473 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_sopFactorRec(ctx ,lr_pointer_cast(LRSop,cQuo_L472),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); eNodeQuo_L473 = _lr_rhs; _lr_rhs)
    (_lr_rhs=lr_cast(Int32,lr_factor_addNodeAnd(lr_pointer_cast(LRSubg,subg) ,lr_cast(Int32,eNodeDiv_L473),lr_cast(Int32,eNodeQuo_L473))); eNodeAnd_L473 = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSop,cRem_L472)).nCubes) == lr_cast(Int32,0))))
        begin _lr_result=lr_cast(Int32,lr_cast(Int32,eNodeAnd_L473)); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    end
    (_lr_rhs=lr_cast(Int32,lr_sopFactorRec(ctx ,lr_pointer_cast(LRSop,cRem_L472),lr_cast(Int32,nLits),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); eNodeRem_L473 = _lr_rhs; _lr_rhs)
    begin _lr_result=lr_cast(Int32,lr_factor_addNodeOr(lr_pointer_cast(LRSubg,subg) ,lr_cast(Int32,eNodeAnd_L473),lr_cast(Int32,eNodeRem_L473))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); return _lr_result end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base)))
    return nothing
end

# Original sopFactor source device body; candidate, not certified.
@inline function lr_sopFactor(ctx,vCover,nCoverSize,fCompl,vCuts,nVars,vecsMem,subg)::Nothing
    _lr_mem_0=lr_alloc(LRSop)
    sop_L499=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(Int32)
    lit0_L551=_lr_mem_1
    _lr_mem_2=lr_alloc(Int32)
    lit1_L551=_lr_mem_2
    _lr_mem_3=lr_alloc(Int32)
    fCompRoot_L551=_lr_mem_3

    cSop_L499=lr_pointer_cast(LRSop,sop_L499)
    local eRoot_L500
    ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,nVars) < lr_cast(Int32,16))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
    (_lr_rhs=lr_cast(Int32,lr_cast(Int32,nVars)); (lr_pointer_cast(LRSubg,subg)).nSize = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,nCoverSize) == lr_cast(Int32,0))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,fCompl)))
            lr_factor_createConst1(lr_pointer_cast(LRSubg,subg))
        else
            lr_factor_createConst0(lr_pointer_cast(LRSubg,subg))
        end
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    end
    if lr_truth((lr_truth(lr_cast(Bool,(lr_cast(Int32,nCoverSize) == lr_cast(Int32,1)))) && lr_truth(lr_cast(Bool,(lr_cast(UInt32,(lr_pointer_cast(UInt32,vCover))[lr_cast(Int32,0)]) == lr_cast(UInt32,lr_cast(Int32,0)))))))
        if lr_truth(lr_cast(Bool,lr_cast(Int32,fCompl)))
            lr_factor_createConst0(lr_pointer_cast(LRSubg,subg))
        else
            lr_factor_createConst1(lr_pointer_cast(LRSubg,subg))
        end
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    end
    lr_sopCreateInverse(ctx ,lr_pointer_cast(LRSop,cSop_L499),lr_pointer_cast(UInt32,vCover),lr_cast(Int32,nCoverSize),lr_pointer_cast(LRVecsMem,vecsMem))
    (_lr_rhs=lr_cast(Int32,lr_sopFactorRec(ctx ,lr_pointer_cast(LRSop,cSop_L499),lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,nVars))),lr_pointer_cast(LRVecsMem,vecsMem),lr_pointer_cast(LRSubg,subg))); eRoot_L500 = _lr_rhs; _lr_rhs)
    if lr_truth(lr_cast(Bool,(lr_s_AigNodeID(lr_cast(Int32,eRoot_L500)) < lr_cast(Int32,nVars))))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(LRSubg,subg)).nSize) == lr_cast(Int32,nVars))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        lr_factor_createSingleExistingVar(lr_pointer_cast(LRSubg,subg) ,lr_s_AigNodeLitCond(lr_cast(Int32,(lr_pointer_cast(Int32,vCuts))[lr_s_AigNodeID(lr_cast(Int32,eRoot_L500))]),lr_cast(Int32,lr_cast(Bool,(lr_s_AigNodeIsComplement(lr_cast(Int32,eRoot_L500)) != lr_cast(Int32,fCompl))))))
        begin _lr_result=(nothing; nothing); lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base))); return _lr_result end
    end
    if lr_truth(lr_cast(Bool,lr_cast(Int32,fCompl)))
        rootNode_L550=lr_cast(UInt64,lr_cast(UInt64,(lr_pointer_cast(UInt64,(lr_pointer_cast(LRSubg,subg)).pArray))[lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSubg,subg)).nSize) - lr_cast(Int32,1)))]))



        lr_factor_unbindAndNodeKeyFlag(lr_cast(UInt64,rootNode_L550),lit0_L551,lit1_L551,fCompRoot_L551)
        (_lr_rhs=lr_cast(UInt64,lr_factor_formAndNodeKeyFlag(lr_cast(Int32,lit0_L551[0]),lr_cast(Int32,lit1_L551[0]),lr_cast(Int32,(lr_cast(Int32,1) - lr_cast(Int32,fCompRoot_L551[0]))))); (lr_pointer_cast(UInt64,(lr_pointer_cast(LRSubg,subg)).pArray))[lr_cast(Int32,(lr_cast(Int32,(lr_pointer_cast(LRSubg,subg)).nSize) - lr_cast(Int32,1)))] = _lr_rhs; _lr_rhs)
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_2,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_3,:base)))
    return nothing
end

# Original factorFromTruth source device body; candidate, not certified.
@inline function lr_factorFromTruth(ctx,vCuts,vCutRanges,vSubgTable,vSubgLinks,vSubgLens,pSubgTableNext,vTruth,vTruthNeg,vTruthRanges,vTruthElem,nResyn)::Nothing
    _lr_mem_0=lr_alloc(LRVecsMem)
    vecsMem_L285=_lr_mem_0[0]
    _lr_mem_1=lr_alloc(LRSubg)
    subg_L286=_lr_mem_1[0]
    idx_L281=lr_cast(Int32,lr_cast(Int32,lr_cast(UInt32,(lr_cast(UInt32,(lr_cast(UInt32,(Int32(blockIdx().x)-Int32(1))) * lr_cast(UInt32,Int32(blockDim().x)))) + lr_cast(UInt32,(Int32(threadIdx().x)-Int32(1)))))))
    local nVars_L282
    local cutStartIdx_L283
    local cutEndIdx_L283
    local truthStartIdx_L283
    local truthEndIdx_L283
    fNeg_L284=lr_cast(Int32,lr_cast(Int32,0))
    vecsMem_L285.nSize=Int32(0)
    subg_L286.nSize=Int32(0)
    if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L281) < lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,nResyn))))))
        if lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L281) >= lr_cast(Int32,nResyn))))
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(idx_L281 - lr_cast(Int32,nResyn)))); idx_L281 = _lr_rhs; _lr_rhs)
            (_lr_rhs=lr_cast(Int32,lr_cast(Int32,1)); fNeg_L284 = _lr_rhs; _lr_rhs)
        end
        (_lr_rhs=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L281) == lr_cast(Int32,0)))) ? lr_cast(Int32,0) : lr_cast(Int32,(lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,(lr_cast(Int32,idx_L281) - lr_cast(Int32,1)))])))); cutStartIdx_L283 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vCutRanges))[lr_cast(Int32,idx_L281)])); cutEndIdx_L283 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,cutEndIdx_L283) - lr_cast(Int32,cutStartIdx_L283)))); nVars_L282 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,((lr_truth(lr_cast(Bool,(lr_cast(Int32,idx_L281) == lr_cast(Int32,0)))) ? lr_cast(Int32,0) : lr_cast(Int32,(lr_pointer_cast(Int32,vTruthRanges))[lr_cast(Int32,(lr_cast(Int32,idx_L281) - lr_cast(Int32,1)))])))); truthStartIdx_L283 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_pointer_cast(Int32,vTruthRanges))[lr_cast(Int32,idx_L281)])); truthEndIdx_L283 = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_cast(Int32,truthEndIdx_L283) - lr_cast(Int32,truthStartIdx_L283))) == lr_factor_TruthWordNum(lr_cast(Int32,nVars_L282)))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        pTruth_L303=lr_pointer_cast(UInt32,((lr_truth(lr_cast(Bool,lr_cast(Int32,fNeg_L284))) ? lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,vTruthNeg) + lr_cast(Int32,truthStartIdx_L283))) : lr_pointer_cast(UInt32,(lr_pointer_cast(UInt32,vTruth) + lr_cast(Int32,truthStartIdx_L283))))))
        lr_minatoIsop(ctx ,lr_pointer_cast(UInt32,pTruth_L303),lr_cast(Int32,nVars_L282),vecsMem_L285)
        lr_factor___syncthreads()
        lr_sopFactor(ctx ,lr_pointer_cast(UInt32,(vecsMem_L285).pArray),lr_cast(Int32,(vecsMem_L285).nSize),lr_cast(Int32,fNeg_L284),lr_pointer_cast(Int32,(lr_pointer_cast(Int32,vCuts) + lr_cast(Int32,cutStartIdx_L283))),lr_cast(Int32,nVars_L282),vecsMem_L285,subg_L286)
        lr_factor___syncthreads()
        local currRowIdx_L312
        local lastRowIdx_L312
        local columnPtr_L312
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,2) * lr_cast(Int32,idx_L281))) + lr_cast(Int32,fNeg_L284)))); currRowIdx_L312 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L312 = _lr_rhs; _lr_rhs)
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,(lr_cast(Int32,(subg_L286).nSize) - lr_cast(Int32,nVars_L282)))); (lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,currRowIdx_L312)] = _lr_rhs; _lr_rhs)
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLens))[lr_cast(Int32,currRowIdx_L312)]) > lr_cast(Int32,0))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L312)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
        (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L312)] = _lr_rhs; _lr_rhs)
        i_L320=lr_cast(Int32,lr_cast(Int32,nVars_L282))
        while lr_truth(lr_cast(Bool,(lr_cast(Int32,i_L320) < lr_cast(Int32,(subg_L286).nSize))))
            if lr_truth(lr_cast(Bool,(lr_cast(Int32,columnPtr_L312) == lr_cast(Int32,lr_cast(Int32,8)))))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currRowIdx_L312)); lastRowIdx_L312 = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_atomicAdd(lr_pointer_cast(Int32,pSubgTableNext),lr_cast(Int32,1))); currRowIdx_L312 = _lr_rhs; _lr_rhs)
                ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,currRowIdx_L312) < lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,4) * lr_cast(Int32,nResyn))) - lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
                ((lr_truth(lr_cast(Bool,lr_cast(Bool,(lr_cast(Int32,(lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L312)]) == lr_cast(Int32,(-lr_cast(Int32,1))))))) ? (lr_cast(Int32,0); nothing) : (CUDA.@cuassert false)))
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,currRowIdx_L312)); (lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,lastRowIdx_L312)] = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); (lr_pointer_cast(Int32,vSubgLinks))[lr_cast(Int32,currRowIdx_L312)] = _lr_rhs; _lr_rhs)
                (_lr_rhs=lr_cast(Int32,lr_cast(Int32,0)); columnPtr_L312 = _lr_rhs; _lr_rhs)
            end
            (_lr_rhs=lr_cast(UInt64,lr_cast(UInt64,(lr_pointer_cast(UInt64,(subg_L286).pArray))[lr_cast(Int32,i_L320)])); (lr_pointer_cast(UInt64,vSubgTable))[lr_cast(Int32,(lr_cast(Int32,(lr_cast(Int32,currRowIdx_L312) * lr_cast(Int32,lr_cast(Int32,8)))) + ((_lr_old=columnPtr_L312; columnPtr_L312=lr_cast(Int32,(columnPtr_L312+Int32(1))); _lr_old))))] = _lr_rhs; _lr_rhs)
            (_lr_old=i_L320; i_L320=lr_cast(Int32,(i_L320+Int32(1))); _lr_old)
        end
        lr_factor___syncthreads()
    end
    lr_free(Ptr{Cvoid}(getfield(_lr_mem_0,:base))); lr_free(Ptr{Cvoid}(getfield(_lr_mem_1,:base)))
    return nothing
end
