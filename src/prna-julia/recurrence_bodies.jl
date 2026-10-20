# Source-derived PRNA recurrence statements. Explicit GPU partitions/reduction tree are separate.
@inline function calc_hairpin_stack_exterior_multibranch_body(i,d,n,s,bcp,v,x,w5,w3,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((!rna_truth(can_pair(i,j,n,bcp))))))
        return nothing
    end
    vij = RNA_INF
    if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0)))))
        if rna_truth(is_interior(i,j))
            (begin vij = hairpin_loop_energy(s,i,j,d,p); vij end)
        end
        if rna_truth((rna_truth(can_pair((i + Int32(1)),(j - Int32(1)),n,bcp)) && rna_truth((!rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= (Int32(3) - Int32(2))))))))))
            (vij = free_energy_sum(vij,(p.stack[s[i]][s[j]][s[(i + Int32(1))]][s[(j - Int32(1))]] + v[ind((i + Int32(1)),(j - Int32(1)),n)])))
        end
    end
    if rna_truth(is_exterior(i,j))
        (vij = free_energy_sum(vij,((w3[(i + Int32(1))] + w5[(j - Int32(1))]) + terminal_U_penalty(s,i,j,p))))
        if rna_truth((i != (n - Int32(1))))
            (vij = free_energy_sum(vij,((w3[(i + Int32(2))] + w5[(j - Int32(1))]) + dangle_3p_energy(s,i,j,(i + Int32(1)),p))))
        end
        if rna_truth((j != Int32(0)))
            (vij = free_energy_sum(vij,((w3[(i + Int32(1))] + w5[(j - Int32(2))]) + dangle_5p_energy(s,i,j,(j - Int32(1)),p))))
        end
        if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0)))))
            (vij = free_energy_sum(vij,((w3[(i + Int32(2))] + w5[(j - Int32(2))]) + terminal_stack(s,i,j,(i + Int32(1)),(j - Int32(1)),p))))
        end
    end
    if rna_truth((rna_truth((rna_truth((d > ((Int32(2) * Int32(3)) + Int32(3)))) && rna_truth((i != (n - Int32(1)))))) && rna_truth((j != Int32(0)))))
        (vij = free_energy_sum(vij,(((x[ind(rem((d - Int32(2)),Int32(5)),(i + Int32(1)),n)] + terminal_U_penalty(s,i,j,p)) + p.a) + p.c)))
        if rna_truth((i != (n - Int32(2))))
            (vij = free_energy_sum(vij,((((x[ind(rem((d - Int32(3)),Int32(5)),(i + Int32(2)),n)] + dangle_3p_energy(s,i,j,(i + Int32(1)),p)) + p.a) + p.b) + p.c)))
        end
        if rna_truth((j != Int32(1)))
            (vij = free_energy_sum(vij,((((x[ind(rem((d - Int32(3)),Int32(5)),(i + Int32(1)),n)] + dangle_5p_energy(s,i,j,(j - Int32(1)),p)) + p.a) + p.b) + p.c)))
        end
        if rna_truth((rna_truth((i != (n - Int32(2)))) && rna_truth((j != Int32(1)))))
            (vij = free_energy_sum(vij,((((x[ind(rem((d - Int32(4)),Int32(5)),(i + Int32(2)),n)] + terminal_stack_multibranch(s,i,j,(i + Int32(1)),(j - Int32(1)),p)) + p.a) + (Int32(2) * p.b)) + p.c)))
        end
    end
    (begin v[ind(i,j,n)] = vij; v[ind(i,j,n)] end)
    return nothing
end

@inline function calc_internal_body(i,d,n,s,bcp,v,p,tx,ty)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= (Int32(3) + Int32(2)))))))) || rna_truth((!rna_truth(can_pair(i,j,n,bcp))))))
        return RNA_INF
    end
    vij = RNA_INF
    d1start = tx
    d1inc = Int32(8)
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
            if rna_truth(can_pair(ip,jp,n,bcp))
                (vij = free_energy_sum(vij,(internal_loop_energy(s,i,j,ip,jp,d1,d2,p) + v[ind(ip,jp,n)])))
            end
        end
    end
    return vij
end

@inline function calc_coaxial_body(i,d,n,s,bcp,v,y,w5,w3,p,lane)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((!rna_truth(can_pair(i,j,n,bcp))))))
        return RNA_INF
    end
    v1 = rna_invalid(v)
    vij = RNA_INF
    if rna_truth(is_exterior(i,j))
        k = Int32(0)
        kstart = Int32(0)
        (begin kstart = (Int32(0)+lane); kstart end)
        kinc = Int32(128)
        for k in kstart:kinc:((j - Int32(3))-Int32(1))
            if rna_truth((begin v1 = array_val(v,k,(j - Int32(1)),n,bcp); v1 end))
                (vij = free_energy_sum(vij,(((w3[(i + Int32(1))] + w5[(k - Int32(1))]) + coaxial_flush(s,k,(j - Int32(1)),j,i,p)) + rna_deref(v1))))
            end
            if rna_truth(((j - Int32(2)) >= Int32(0)))
                if rna_truth((rna_truth((i < (n - Int32(1)))) && rna_truth((begin v1 = array_val(v,k,(j - Int32(2)),n,bcp); v1 end))))
                    (vij = free_energy_sum(vij,(((w3[(i + Int32(2))] + w5[(k - Int32(1))]) + coaxial_mismatch2(s,k,(j - Int32(2)),j,i,p)) + rna_deref(v1))))
                end
                if rna_truth((begin v1 = array_val(v,(k + Int32(1)),(j - Int32(2)),n,bcp); v1 end))
                    (vij = free_energy_sum(vij,(((w3[(i + Int32(1))] + w5[(k - Int32(1))]) + coaxial_mismatch1(s,(k + Int32(1)),(j - Int32(2)),j,i,p)) + rna_deref(v1))))
                end
            end
        end
        (begin kstart = (((i + Int32(3)) + Int32(1))+lane); kstart end)
        for k in kstart:kinc:(n-Int32(1))
            if rna_truth((begin v1 = array_val(v,(i + Int32(1)),k,n,bcp); v1 end))
                (vij = free_energy_sum(vij,(((w3[(k + Int32(1))] + w5[(j - Int32(1))]) + coaxial_flush(s,j,i,(i + Int32(1)),k,p)) + rna_deref(v1))))
            end
            if rna_truth((rna_truth((j > Int32(0))) && rna_truth((begin v1 = array_val(v,(i + Int32(2)),k,n,bcp); v1 end))))
                (vij = free_energy_sum(vij,(((w3[(k + Int32(1))] + w5[(j - Int32(2))]) + coaxial_mismatch1(s,j,i,(i + Int32(2)),k,p)) + rna_deref(v1))))
            end
            if rna_truth((begin v1 = array_val(v,(i + Int32(2)),(k - Int32(1)),n,bcp); v1 end))
                (vij = free_energy_sum(vij,(((w3[(k + Int32(1))] + w5[(j - Int32(1))]) + coaxial_mismatch2(s,j,i,(i + Int32(2)),(k - Int32(1)),p)) + rna_deref(v1))))
            end
        end
    end
    if rna_truth((rna_truth((rna_truth((d > ((Int32(2) * Int32(3)) + Int32(3)))) && rna_truth((i != (n - Int32(1)))))) && rna_truth((j != Int32(0)))))
        ktmp = Int32(0)
        ktmpstart = ((i + Int32(2))+lane)
        ktmpinc = Int32(128)
        for ktmp in ktmpstart:ktmpinc:((jtmp - Int32(2))-Int32(1))
            k = wrap(ktmp,n)
            if rna_truth((k != (n - Int32(1))))
                if rna_truth((begin v1 = array_val(v,(i + Int32(1)),k,n,bcp); v1 end))
                    (vij = free_energy_sum(vij,(((coaxial_flush(s,j,i,(i + Int32(1)),k,p) + rna_deref(v1)) + p.a_2c) + y[ind((k + Int32(1)),(j - Int32(1)),n)])))
                end
                if rna_truth((rna_truth((rna_truth((rna_truth(((ktmp + Int32(2)) < (jtmp - Int32(1)))) && rna_truth(((i + Int32(1)) != (n - Int32(1)))))) && rna_truth(((k + Int32(1)) != (n - Int32(1)))))) && rna_truth((begin v1 = array_val(v,(i + Int32(2)),k,n,bcp); v1 end))))
                    tmp = (rna_deref(v1) + p.a_2b_2c)
                    (vij = free_energy_sum(vij,((coaxial_mismatch2(s,j,i,(i + Int32(2)),k,p) + tmp) + y[ind((k + Int32(2)),(j - Int32(1)),n)])))
                    if rna_truth((j != Int32(1)))
                        (vij = free_energy_sum(vij,((coaxial_mismatch1(s,j,i,(i + Int32(2)),k,p) + tmp) + y[ind((k + Int32(1)),(j - Int32(2)),n)])))
                    end
                end
            end
        end
        (begin ktmpstart = ((i + Int32(3))+lane); ktmpstart end)
        for ktmp in ktmpstart:ktmpinc:((jtmp - Int32(1))-Int32(1))
            k = wrap(ktmp,n)
            if rna_truth((k != Int32(0)))
                if rna_truth((begin v1 = array_val(v,k,(j - Int32(1)),n,bcp); v1 end))
                    (vij = free_energy_sum(vij,(((coaxial_flush(s,k,(j - Int32(1)),j,i,p) + rna_deref(v1)) + p.a_2c) + y[ind((i + Int32(1)),(k - Int32(1)),n)])))
                end
                if rna_truth((rna_truth((rna_truth((j != Int32(1))) && rna_truth((ktmp > (i + Int32(3)))))) && rna_truth((begin v1 = array_val(v,k,(j - Int32(2)),n,bcp); v1 end))))
                    tmp = (rna_deref(v1) + p.a_2b_2c)
                    if rna_truth((k != Int32(1)))
                        (vij = free_energy_sum(vij,((coaxial_mismatch1(s,k,(j - Int32(2)),j,i,p) + tmp) + y[ind((i + Int32(1)),(k - Int32(2)),n)])))
                    end
                    if rna_truth((i != (n - Int32(2))))
                        (vij = free_energy_sum(vij,((coaxial_mismatch2(s,k,(j - Int32(2)),j,i,p) + tmp) + y[ind((i + Int32(2)),(k - Int32(1)),n)])))
                    end
                end
            end
        end
    end
    return vij
end

@inline function calc_wl_body(i,d,n,s,bcp,v,z,wq,w,wl,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3)))))
        return nothing
    end
    wqtmp = RNA_INF
    wltmp = RNA_INF
    v1 = rna_invalid(v)
    if rna_truth((begin v1 = array_val(v,i,j,n,bcp); v1 end))
        tmp = (rna_deref(v1) + terminal_U_penalty(s,i,j,p))
        (wqtmp = free_energy_sum(wqtmp,tmp))
        (wltmp = free_energy_sum(wltmp,(tmp + p.c)))
    end
    if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((begin v1 = array_val(v,(i + Int32(1)),j,n,bcp); v1 end))))
        tmp = (rna_deref(v1) + dangle_5p_energy(s,j,(i + Int32(1)),i,p))
        (wqtmp = free_energy_sum(wqtmp,tmp))
        (wltmp = free_energy_sum(wltmp,((tmp + p.b) + p.c)))
    end
    if rna_truth((rna_truth((j != Int32(0))) && rna_truth((begin v1 = array_val(v,i,(j - Int32(1)),n,bcp); v1 end))))
        tmp = (rna_deref(v1) + dangle_3p_energy(s,(j - Int32(1)),i,j,p))
        (wqtmp = free_energy_sum(wqtmp,tmp))
        (wltmp = free_energy_sum(wltmp,((tmp + p.b) + p.c)))
    end
    if rna_truth((rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((j != Int32(0))))) && rna_truth((begin v1 = array_val(v,(i + Int32(1)),(j - Int32(1)),n,bcp); v1 end))))
        tmp = (rna_deref(v1) + terminal_stack_multibranch(s,(j - Int32(1)),(i + Int32(1)),j,i,p))
        (wqtmp = free_energy_sum(wqtmp,tmp))
        (wltmp = free_energy_sum(wltmp,((tmp + (Int32(2) * p.b)) + p.c)))
    end
    if rna_truth(is_interior(i,j))
        (begin wq[upper_triangle_index(i,j)] = wqtmp; wq[upper_triangle_index(i,j)] end)
    end
    (begin wl[ind(rem(d,Int32(2)),i,n)] = (begin z[ind(i,j,n)] = wltmp; z[ind(i,j,n)] end); wl[ind(rem(d,Int32(2)),i,n)] end)
    if rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((d > Int32(0)))))
        (wl[ind(rem(d,Int32(2)),i,n)] = free_energy_sum(wl[ind(rem(d,Int32(2)),i,n)],(wl[ind(rem((d - Int32(1)),Int32(2)),(i + Int32(1)),n)] + p.b)))
    end
    (begin w[ind(rem(d,Int32(2)),i,n)] = wl[ind(rem(d,Int32(2)),i,n)]; w[ind(rem(d,Int32(2)),i,n)] end)
    if rna_truth((rna_truth((j != Int32(0))) && rna_truth((d > Int32(0)))))
        (w[ind(rem(d,Int32(2)),i,n)] = free_energy_sum(w[ind(rem(d,Int32(2)),i,n)],(w[ind(rem((d - Int32(1)),Int32(2)),i,n)] + p.b)))
    end
    return nothing
end

@inline function calc_xl_body(i,d,n,z,yl,xl,lane)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3)))))
        return RNA_INF
    end
    if rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= ((Int32(2) * Int32(3)) + Int32(1))))))
        return RNA_INF
    end
    kstart = ((i + Int32(1))+lane)
    kinc = Int32(128)
    ktmp = Int32(0)
    tmp = RNA_INF
    for ktmp in kstart:kinc:((jtmp - Int32(1))-Int32(1))
        if rna_truth((ktmp != (n - Int32(1))))
            k = wrap(ktmp,n)
            (tmp = free_energy_sum(tmp,(z[ind(i,k,n)] + yl[ind((k + Int32(1)),j,n)])))
        end
    end
    return tmp
end

@inline function calc_z_body(i,d,n,s,bcp,v,z,xl,wq,p,lane)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3))))) || rna_truth((rna_truth(is_interior(i,j)) && rna_truth((d <= ((Int32(2) * Int32(3)) + Int32(1))))))))
        return (RNA_INF,RNA_INF)
    end
    kstart = (((i + Int32(3)) + Int32(1))+lane)
    kinc = Int32(128)
    ktmp = Int32(0)
    tmp1 = RNA_INF
    tmp2 = RNA_INF
    for ktmp in kstart:kinc:(((jtmp - Int32(3)) - Int32(1))-Int32(1))
        k = wrap(ktmp,n)
        if rna_truth((k == (n - Int32(1))))
            continue
        end
        v1 = rna_invalid(v)
        v2 = rna_invalid(v)
        if rna_truth((rna_truth((begin v1 = array_val(v,i,k,n,bcp); v1 end)) && rna_truth((begin v2 = array_val(v,(k + Int32(1)),j,n,bcp); v2 end))))
            (tmp1 = free_energy_sum(tmp1,((rna_deref(v1) + rna_deref(v2)) + coaxial_flush(s,i,k,(k + Int32(1)),j,p))))
        end
        if rna_truth((rna_truth((j == Int32(0))) || rna_truth(((k + Int32(1)) == (n - Int32(1))))))
            continue
        end
        if rna_truth((rna_truth((rna_truth((i != (n - Int32(1)))) && rna_truth((begin v1 = array_val(v,(i + Int32(1)),k,n,bcp); v1 end)))) && rna_truth((begin v2 = array_val(v,(k + Int32(2)),j,n,bcp); v2 end))))
            (tmp2 = free_energy_sum(tmp2,((rna_deref(v1) + rna_deref(v2)) + coaxial_mismatch1(s,(i + Int32(1)),k,(k + Int32(2)),j,p))))
        end
        if rna_truth((rna_truth((begin v1 = array_val(v,i,k,n,bcp); v1 end)) && rna_truth((begin v2 = array_val(v,(k + Int32(2)),(j - Int32(1)),n,bcp); v2 end))))
            (tmp2 = free_energy_sum(tmp2,((rna_deref(v1) + rna_deref(v2)) + coaxial_mismatch2(s,i,k,(k + Int32(2)),(j - Int32(1)),p))))
        end
    end
    return tmp1,tmp2
end

@inline function calc_x_body(i,d,n,yl,y,w,wl,xl,x,p)
    jtmp = ((i + d) + Int32(1))
    j = wrap(jtmp,n)
    if rna_truth((rna_truth(is_exterior(i,j)) && rna_truth(((i - j) <= Int32(3)))))
        return nothing
    end
    (begin x[ind(rem(d,Int32(5)),i,n)] = RNA_INF; x[ind(rem(d,Int32(5)),i,n)] end)
    if rna_truth((rna_truth((d > ((Int32(2) * Int32(3)) + Int32(1)))) || rna_truth(is_exterior(i,j))))
        if rna_truth((i != (n - Int32(1))))
            (xl[ind(rem(d,Int32(2)),i,n)] = free_energy_sum(xl[ind(rem(d,Int32(2)),i,n)],(xl[ind(rem((d - Int32(1)),Int32(2)),(i + Int32(1)),n)] + p.b)))
        end
        (begin x[ind(rem(d,Int32(5)),i,n)] = xl[ind(rem(d,Int32(2)),i,n)]; x[ind(rem(d,Int32(5)),i,n)] end)
        if rna_truth((j != Int32(0)))
            (x[ind(rem(d,Int32(5)),i,n)] = free_energy_sum(x[ind(rem(d,Int32(5)),i,n)],(x[ind(rem((d - Int32(1)),Int32(5)),i,n)] + p.b)))
        end
    end
    (begin yl[ind(i,j,n)] = free_energy_sum(wl[ind(rem(d,Int32(2)),i,n)],xl[ind(rem(d,Int32(2)),i,n)]); yl[ind(i,j,n)] end)
    (begin y[ind(i,j,n)] = free_energy_sum(w[ind(rem(d,Int32(2)),i,n)],x[ind(rem(d,Int32(5)),i,n)]); y[ind(i,j,n)] end)
    return nothing
end

@inline function calc_w5_and_w3_body(d,n,w5,w3,wq,lane)
    istart = lane
    iinc = Int32(128)
    w5tmp = RNA_INF
    w3tmp = RNA_INF
    i = Int32(0)
    for i in istart:iinc:(d-Int32(3))
        (w5tmp = free_energy_sum(w5tmp,(w5[(i - Int32(1))] + wq[upper_triangle_index(i,(d + Int32(1)))])))
        (w3tmp = free_energy_sum(w3tmp,(w3[(n - i)] + wq[upper_triangle_index(((n - d) - Int32(2)),((n - i) - Int32(1)))])))
    end
    return w5tmp,w3tmp
end

