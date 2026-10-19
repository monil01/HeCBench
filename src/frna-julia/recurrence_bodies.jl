# Generated recurrence statements from exact native source; reviewed per-cell/partial mappings.
# See generate_device_bodies.py and native_kernels.cuh; GPL-2.0-or-later original source retained.
@inline function calc_V_hairpin_and_V_stack_body(i,d,n,s,v,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_interior(i,j)) && rna_truth((!rna_truth(can_pair(i,j,n,s)))))) || rna_truth((rna_truth(is_exterior(i,j)) && rna_truth((!rna_truth(is_canonical_pair(s[i],s[j]))))))))
        (begin v[ind(i,j,n)] = Int32(14000); v[ind(i,j,n)] end)
        return nothing
    end
    vij = Int32(14000)
    if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0)))))
        if rna_truth(is_interior(i,j))
            (begin vij = hairpin_loop_energy(s,i,j,d,p); vij end)
        end
        if rna_truth((rna_truth(can_pair((i + Int32(1)),(j - Int32(1)),n,s)) && rna_truth((!rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= (Int32(3) - Int32(2))))))))))
            (vij = min(vij,(p.stack[s[i]][s[j]][s[(i + Int32(1))]][s[(j - Int32(1))]] + v[ind((i + Int32(1)),(j - Int32(1)),n)])))
        end
    end
    (begin v[ind(i,j,n)] = vij; v[ind(i,j,n)] end)
    return nothing
end

@inline function calc_V_bulge_internal_body(i,d,n,s,v,p,tx,ty)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= (Int32(3) + Int32(2)))))))) || rna_truth((!rna_truth(can_pair(i,j,n,s))))))
        return RNA_INF
    end
    vij = Int32(14000)
    d1start = tx
    d1inc = Int32(16)
    dmax = min(Int32(30),(d - Int32(2)))
    d1max = min(dmax,((n - i) - Int32(2)))
    d1 = Int32(0)
    for d1 in d1start:d1inc:d1max
        ip = ((i + d1) + Int32(1))
        d2max = min((dmax - d1),(j - Int32(1)))
        d2start = (d1>0 ? ty : ty+Int32(1))
        d2inc = Int32(16)
        d2 = Int32(0)
        for d2 in d2start:d2inc:d2max
            jp = ((j - d2) - Int32(1))
            if rna_truth(can_pair(ip,jp,n,s))
                (vij = min(vij,(internal_loop_energy(s,n,i,j,ip,jp,d1,d2,p) + v[ind(ip,jp,n)])))
            end
        end
    end
    return vij
end

@inline function calc_V_multibranch_body(i,d,n,s,v,wm,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((!rna_truth(can_pair(i,j,n,s))))))
        return nothing
    end
    vij = Int32(14000)
    if rna_truth((rna_truth((rna_truth((d > ((Int32(2) * Int32(3)) + Int32(3)))) && rna_truth((i != (n - Int32(1)))))) && rna_truth((j != Int32(0)))))
        (vij = min(vij,(((wm[ind((i + Int32(1)),(j - Int32(1)),n)] + terminal_U_penalty(s,i,j,p)) + p.a) + p.c)))
        if rna_truth((i != (n - Int32(2))))
            (vij = min(vij,((((wm[ind((i + Int32(2)),(j - Int32(1)),n)] + dangle_3p_energy(s,i,j,(i + Int32(1)),p)) + p.a) + p.b) + p.c)))
        end
        if rna_truth((j != Int32(1)))
            (vij = min(vij,((((wm[ind((i + Int32(1)),(j - Int32(2)),n)] + dangle_5p_energy(s,i,j,(j - Int32(1)),p)) + p.a) + p.b) + p.c)))
        end
        if rna_truth((rna_truth((i != (n - Int32(2)))) && rna_truth((j != Int32(1)))))
            (vij = min(vij,((((wm[ind((i + Int32(2)),(j - Int32(2)),n)] + terminal_stack_multibranch(s,i,j,(i + Int32(1)),(j - Int32(1)),p)) + p.a) + (Int32(2) * p.b)) + p.c)))
        end
    end
    (v[ind(i,j,n)] = min(v[ind(i,j,n)],vij))
    return nothing
end

@inline function calc_V_exterior_body(i,d,n,s,v,w5,w3,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth(is_interior(i,j))
        return nothing
    end
    vij = Int32(14000)
    if rna_truth((rna_truth(is_canonical_pair(s[i],s[j])) && rna_truth(not_isolated(i,j,n,s))))
        (vij = min(vij,((w3[(i + Int32(1))] + w5[(j - Int32(1))]) + terminal_U_penalty(s,i,j,p))))
        if rna_truth((i != (n - Int32(1))))
            (vij = min(vij,((w3[(i + Int32(2))] + w5[(j - Int32(1))]) + dangle_3p_energy(s,i,j,(i + Int32(1)),p))))
        end
        if rna_truth((j != Int32(0)))
            (vij = min(vij,((w3[(i + Int32(1))] + w5[(j - Int32(2))]) + dangle_5p_energy(s,i,j,(j - Int32(1)),p))))
        end
        if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0)))))
            (vij = min(vij,((w3[(i + Int32(2))] + w5[(j - Int32(2))]) + terminal_stack(s,i,j,(i + Int32(1)),(j - Int32(1)),p))))
        end
    end
    (v[ind(i,j,n)] = min(v[ind(i,j,n)],vij))
    return nothing
end

@inline function calc_W_body(i,d,n,s,v,w,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    wij = Int32(14000)
    v_temp = rna_invalid(v)
    if rna_truth((d > Int32(0)))
        if rna_truth((i != (n - Int32(1))))
            (wij = min(wij,(w[ind((i + Int32(1)),j,n)] + p.b)))
        end
        if rna_truth((j != Int32(0)))
            (wij = min(wij,(w[ind(i,(j - Int32(1)),n)] + p.b)))
        end
    end
    if rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d > (Int32(3) - Int32(1))))))
        (begin v_temp = array_val(v,i,j,n,s); v_temp end)
        (wij = min(wij,(((rna_truth(v_temp) ? rna_deref(v_temp) : Int32(14000)) + terminal_U_penalty(s,i,j,p)) + p.c)))
        if rna_truth((j != Int32(0)))
            (begin v_temp = array_val(v,i,(j - Int32(1)),n,s); v_temp end)
            (wij = min(wij,((((rna_truth(v_temp) ? rna_deref(v_temp) : Int32(14000)) + dangle_3p_energy(s,(j - Int32(1)),i,j,p)) + p.b) + p.c)))
        end
        if rna_truth((i != (n - Int32(1))))
            (begin v_temp = array_val(v,(i + Int32(1)),j,n,s); v_temp end)
            (wij = min(wij,((((rna_truth(v_temp) ? rna_deref(v_temp) : Int32(14000)) + dangle_5p_energy(s,j,(i + Int32(1)),i,p)) + p.b) + p.c)))
        end
        if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0)))))
            (begin v_temp = array_val(v,(i + Int32(1)),(j - Int32(1)),n,s); v_temp end)
            (wij = min(wij,((((rna_truth(v_temp) ? rna_deref(v_temp) : Int32(14000)) + terminal_stack_multibranch(s,(j - Int32(1)),(i + Int32(1)),j,i,p)) + (Int32(2) * p.b)) + p.c)))
        end
    end
    if rna_truth(is_exterior(i,j))
        (wij = min(wij,((v[ind(i,j,n)] + terminal_U_penalty(s,i,j,p)) + p.c)))
        if rna_truth((j != Int32(0)))
            (wij = min(wij,(((v[ind(i,(j - Int32(1)),n)] + dangle_3p_energy(s,(j - Int32(1)),i,j,p)) + p.b) + p.c)))
        end
        if rna_truth((i != (n - Int32(1))))
            (wij = min(wij,(((v[ind((i + Int32(1)),j,n)] + dangle_5p_energy(s,j,(i + Int32(1)),i,p)) + p.b) + p.c)))
        end
        if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0)))))
            (wij = min(wij,(((v[ind((i + Int32(1)),(j - Int32(1)),n)] + terminal_stack_multibranch(s,(j - Int32(1)),(i + Int32(1)),j,i,p)) + (Int32(2) * p.b)) + p.c)))
        end
    end
    (begin w[ind(i,j,n)] = wij; w[ind(i,j,n)] end)
    return nothing
end

@inline function calc_WM_body(i,d,n,s,w,wm,p,lane)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    tmp = Int32(14000)
    if rna_truth((rna_truth(is_interior(i,j)) && rna_truth((((j - i) - Int32(1)) <= ((Int32(2) * Int32(3)) + Int32(2))))))
        (begin wm[ind(i,j,n)] = Int32(14000); wm[ind(i,j,n)] end)
        return (RNA_INF,false)
    end
    kstart = i+lane
    kinc = Int32(256)
    ktmp = Int32(0)
    for ktmp in kstart:kinc:(jtmp-Int32(1))
        if rna_truth((ktmp != (n - Int32(1))))
            k = wrap(ktmp,n)
            (tmp = min(tmp,(w[ind(i,k,n)] + w[ind((k + Int32(1)),j,n)])))
        end
    end
    if rna_truth((d > Int32(0)))
        if rna_truth((i != (n - Int32(1))))
            (tmp = min(tmp,(wm[ind((i + Int32(1)),j,n)] + p.b)))
        end
        if rna_truth((j != Int32(0)))
            (tmp = min(tmp,(wm[ind(i,(j - Int32(1)),n)] + p.b)))
        end
    end
    return tmp,true
end

@inline function calc_coaxial_body(i,d,n,s,v,w,w5,w3,p,lane)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((!rna_truth(can_pair(i,j,n,s))))))
        return RNA_INF
    end
    v1 = rna_invalid(v)
    vij = Int32(14000)
    if rna_truth(is_exterior(i,j))
        k = Int32(0)
        kstart = lane
        (begin kstart = (Int32(0)+lane); kstart end)
        kinc = Int32(256)
        for k in kstart:kinc:((j - Int32(3))-Int32(1))
            if rna_truth((begin v1 = array_val(v,k,(j - Int32(1)),n,s); v1 end))
                (vij = min(vij,(((w3[(i + Int32(1))] + w5[(k - Int32(1))]) + coaxial_flush(s,k,(j - Int32(1)),j,i,p)) + rna_deref(v1))))
            end
            if rna_truth(((j - Int32(2)) >= Int32(0)))
                if rna_truth((rna_truth((i < (n - Int32(1)))) && rna_truth((begin v1 = array_val(v,k,(j - Int32(2)),n,s); v1 end))))
                    (vij = min(vij,(((w3[(i + Int32(2))] + w5[(k - Int32(1))]) + coaxial_mismatch2(s,k,(j - Int32(2)),j,i,p)) + rna_deref(v1))))
                end
                if rna_truth((begin v1 = array_val(v,(k + Int32(1)),(j - Int32(2)),n,s); v1 end))
                    (vij = min(vij,(((w3[(i + Int32(1))] + w5[(k - Int32(1))]) + coaxial_mismatch1(s,(k + Int32(1)),(j - Int32(2)),j,i,p)) + rna_deref(v1))))
                end
            end
        end
        (begin kstart = (((i + Int32(3)) + Int32(1))+lane); kstart end)
        for k in kstart:kinc:(n-Int32(1))
            if rna_truth((begin v1 = array_val(v,(i + Int32(1)),k,n,s); v1 end))
                (vij = min(vij,(((w3[(k + Int32(1))] + w5[(j - Int32(1))]) + coaxial_flush(s,j,i,(i + Int32(1)),k,p)) + rna_deref(v1))))
            end
            if rna_truth((rna_truth((j > Int32(0))) && rna_truth((begin v1 = array_val(v,(i + Int32(2)),k,n,s); v1 end))))
                (vij = min(vij,(((w3[(k + Int32(1))] + w5[(j - Int32(2))]) + coaxial_mismatch1(s,j,i,(i + Int32(2)),k,p)) + rna_deref(v1))))
            end
            if rna_truth((begin v1 = array_val(v,(i + Int32(2)),(k - Int32(1)),n,s); v1 end))
                (vij = min(vij,(((w3[(k + Int32(1))] + w5[(j - Int32(1))]) + coaxial_mismatch2(s,j,i,(i + Int32(2)),(k - Int32(1)),p)) + rna_deref(v1))))
            end
        end
    end
    if rna_truth((rna_truth((rna_truth((d > ((Int32(2) * Int32(3)) + Int32(3)))) && rna_truth((i != (n - Int32(1)))))) && rna_truth((j != Int32(0)))))
        ktmp = Int32(0)
        ktmpstart = ((i + Int32(2))+lane)
        ktmpinc = Int32(256)
        for ktmp in ktmpstart:ktmpinc:((jtmp - Int32(2))-Int32(1))
            k = wrap(ktmp,n)
            if rna_truth((k != (n - Int32(1))))
                if rna_truth((begin v1 = array_val(v,(i + Int32(1)),k,n,s); v1 end))
                    (vij = min(vij,(((coaxial_flush(s,j,i,(i + Int32(1)),k,p) + rna_deref(v1)) + p.a_2c) + w[ind((k + Int32(1)),(j - Int32(1)),n)])))
                end
                if rna_truth((rna_truth((rna_truth((rna_truth(((ktmp + Int32(2)) < (jtmp - Int32(1)))) && rna_truth(((i + Int32(1)) != (n - Int32(1)))))) && rna_truth(((k + Int32(1)) != (n - Int32(1)))))) && rna_truth((begin v1 = array_val(v,(i + Int32(2)),k,n,s); v1 end))))
                    tmp = (rna_deref(v1) + p.a_2b_2c)
                    (vij = min(vij,((coaxial_mismatch2(s,j,i,(i + Int32(2)),k,p) + tmp) + w[ind((k + Int32(2)),(j - Int32(1)),n)])))
                    if rna_truth((j != Int32(1)))
                        (vij = min(vij,((coaxial_mismatch1(s,j,i,(i + Int32(2)),k,p) + tmp) + w[ind((k + Int32(1)),(j - Int32(2)),n)])))
                    end
                end
            end
        end
        (begin ktmpstart = ((i + Int32(3))+lane); ktmpstart end)
        for ktmp in ktmpstart:ktmpinc:((jtmp - Int32(1))-Int32(1))
            k = wrap(ktmp,n)
            if rna_truth((k != Int32(0)))
                if rna_truth((begin v1 = array_val(v,k,(j - Int32(1)),n,s); v1 end))
                    (vij = min(vij,(((coaxial_flush(s,k,(j - Int32(1)),j,i,p) + rna_deref(v1)) + p.a_2c) + w[ind((i + Int32(1)),(k - Int32(1)),n)])))
                end
                if rna_truth((rna_truth((rna_truth((j != Int32(1))) && rna_truth((ktmp > (i + Int32(3)))))) && rna_truth((begin v1 = array_val(v,k,(j - Int32(2)),n,s); v1 end))))
                    tmp = (rna_deref(v1) + p.a_2b_2c)
                    if rna_truth((k != Int32(1)))
                        (vij = min(vij,((coaxial_mismatch1(s,k,(j - Int32(2)),j,i,p) + tmp) + w[ind((i + Int32(1)),(k - Int32(2)),n)])))
                    end
                    if rna_truth((i != (n - Int32(2))))
                        (vij = min(vij,((coaxial_mismatch2(s,k,(j - Int32(2)),j,i,p) + tmp) + w[ind((i + Int32(2)),(k - Int32(1)),n)])))
                    end
                end
            end
        end
    end
    return vij
end

@inline function calc_wl_coax_body(i,d,n,s,v,w,wm,p,wca,lane)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= ((Int32(2) * Int32(3)) + Int32(1))))))))
        return (RNA_INF,RNA_INF,false)
    end
    kstart = i+Int32(4)+lane
    kinc = Int32(256)
    ktmp = Int32(0)
    tmp1 = Int32(14000)
    tmp2 = Int32(14000)
    for ktmp in kstart:kinc:(((jtmp - Int32(3)) - Int32(1))-Int32(1))
        k = wrap(ktmp,n)
        if rna_truth((k == (n - Int32(1))))
            continue
        end
        v1 = rna_invalid(v)
        v2 = rna_invalid(v)
        if rna_truth((rna_truth((begin v1 = array_val(v,i,k,n,s); v1 end)) && rna_truth((begin v2 = array_val(v,(k + Int32(1)),j,n,s); v2 end))))
            (tmp1 = min(tmp1,((rna_deref(v1) + rna_deref(v2)) + coaxial_flush(s,i,k,(k + Int32(1)),j,p))))
        end
        if rna_truth((rna_truth((j == Int32(0))) || rna_truth(((k + Int32(1)) == (n - Int32(1))))))
            continue
        end
        if rna_truth((rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((begin v1 = array_val(v,(i + Int32(1)),k,n,s); v1 end)))) && rna_truth((begin v2 = array_val(v,(k + Int32(2)),j,n,s); v2 end))))
            (tmp2 = min(tmp2,((rna_deref(v1) + rna_deref(v2)) + coaxial_mismatch1(s,(i + Int32(1)),k,(k + Int32(2)),j,p))))
        end
        if rna_truth((rna_truth((begin v1 = array_val(v,i,k,n,s); v1 end)) && rna_truth((begin v2 = array_val(v,(k + Int32(2)),(j - Int32(1)),n,s); v2 end))))
            (tmp2 = min(tmp2,((rna_deref(v1) + rna_deref(v2)) + coaxial_mismatch2(s,i,k,(k + Int32(2)),(j - Int32(1)),p))))
        end
    end
    return tmp1,tmp2,true
end

@inline function calc_w5_and_w3_body(d,n,s,v,w5,w3,p,wca,lane)
    istart = lane
    iinc = Int32(256)
    w5tmp = Int32(0)
    w3tmp = Int32(0)
    i = Int32(0)
    v_temp = rna_invalid(v)
    for i in istart:iinc:(d-Int32(3))
        if rna_truth((begin v_temp = array_val(v,i,(d + Int32(1)),n,s); v_temp end))
            (w5tmp = min(w5tmp,((w5[(i - Int32(1))] + rna_deref(v_temp)) + terminal_U_penalty(s,(d + Int32(1)),i,p))))
        end
        if rna_truth(((d - i) > Int32(3)))
            if rna_truth((begin v_temp = array_val(v,i,d,n,s); v_temp end))
                (w5tmp = min(w5tmp,((w5[(i - Int32(1))] + rna_deref(v_temp)) + dangle_3p_energy(s,d,i,(d + Int32(1)),p))))
            end
            if rna_truth((begin v_temp = array_val(v,(i + Int32(1)),(d + Int32(1)),n,s); v_temp end))
                (w5tmp = min(w5tmp,((w5[(i - Int32(1))] + rna_deref(v_temp)) + dangle_5p_energy(s,(d + Int32(1)),(i + Int32(1)),i,p))))
            end
            (w5tmp = min(w5tmp,(w5[(i - Int32(1))] + wca[ind(i,(d + Int32(1)),n)])))
        end
        if rna_truth((rna_truth(((d - i) > (Int32(3) + Int32(1)))) && rna_truth((begin v_temp = array_val(v,(i + Int32(1)),d,n,s); v_temp end))))
            (w5tmp = min(w5tmp,((w5[(i - Int32(1))] + rna_deref(v_temp)) + terminal_stack(s,d,(i + Int32(1)),(d + Int32(1)),i,p))))
        end
        if rna_truth((begin v_temp = array_val(v,((n - d) - Int32(2)),((n - i) - Int32(1)),n,s); v_temp end))
            (w3tmp = min(w3tmp,((w3[(n - i)] + rna_deref(v_temp)) + terminal_U_penalty(s,((n - i) - Int32(1)),((n - d) - Int32(2)),p))))
        end
        if rna_truth((begin v_temp = array_val(v,((n - d) - Int32(2)),((n - i) - Int32(2)),n,s); v_temp end))
            (w3tmp = min(w3tmp,((w3[(n - i)] + rna_deref(v_temp)) + dangle_3p_energy(s,((n - i) - Int32(2)),((n - d) - Int32(2)),((n - i) - Int32(1)),p))))
        end
        if rna_truth((rna_truth((((n - d) - Int32(1)) != Int32(0))) && rna_truth((begin v_temp = array_val(v,((n - d) - Int32(1)),((n - i) - Int32(1)),n,s); v_temp end))))
            (w3tmp = min(w3tmp,((w3[(n - i)] + rna_deref(v_temp)) + dangle_5p_energy(s,((n - i) - Int32(1)),((n - d) - Int32(1)),((n - d) - Int32(2)),p))))
        end
        if rna_truth((rna_truth((rna_truth((((n - i) - Int32(2)) != (n - Int32(1)))) && rna_truth((((n - d) - Int32(1)) != Int32(0))))) && rna_truth((begin v_temp = array_val(v,((n - d) - Int32(1)),((n - i) - Int32(2)),n,s); v_temp end))))
            (w3tmp = min(w3tmp,((w3[(n - i)] + rna_deref(v_temp)) + terminal_stack(s,((n - i) - Int32(2)),((n - d) - Int32(1)),((n - i) - Int32(1)),((n - d) - Int32(2)),p))))
        end
        (w3tmp = min(w3tmp,(w3[(n - i)] + wca[ind(((n - d) - Int32(2)),((n - i) - Int32(1)),n)])))
    end
    return w5tmp,w3tmp
end

