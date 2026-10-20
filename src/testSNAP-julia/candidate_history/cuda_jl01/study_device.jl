# Derived from Sandia TestSNAP Zero-Clause BSD source. See original_native/ attribution.
using StaticArrays
struct SnapShift{A}
    array::A
    offset::Int32
end
@inline Base.getindex(s::SnapShift,i::Integer)=s.array[s.offset+i]
@inline function compute_sfac(r,rcut,switch_flag)
    @inbounds begin
    if (switch_flag == Int32(0))
        return 1.0
    end
    if (switch_flag == Int32(1))
        if (r <= 0.0)
            return 1.0
        else
            if (r > rcut)
                return 0.0
            else
                rcutfac_s1 = (3.14159265358979323846 / (rcut - 0.0))
                return (0.5 * (cos(((r - 0.0) * rcutfac_s1)) + 1.0))
            end
        end
    end
    return 0.0
    end
    return 0.0
end

@inline function compute_dsfac(r,rcut,switch_flag)
    @inbounds begin
    if (switch_flag == Int32(0))
        return 0.0
    end
    if (switch_flag == Int32(1))
        if (r <= 0.0)
            return 0.0
        else
            if (r > rcut)
                return 0.0
            else
                rcutfac_s1 = (3.14159265358979323846 / (rcut - 0.0))
                return (((-0.5) * sin(((r - 0.0) * rcutfac_s1))) * rcutfac_s1)
            end
        end
    end
    return 0.0
    end
    return 0.0
end

@inline function compute_duarray(natom,nbor,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag,x,y,z,z0,r,dz0dr,wj_in,rcut,rootpqarray,ulist,dulist)
    @inbounds begin
    local r0inv_s1
    local a_r_s2
    local a_i_s3
    local b_r_s4
    local b_i_s5
    da_r_s6 = MVector{3,Float64}(undef)
    da_i_s7 = MVector{3,Float64}(undef)
    db_r_s8 = MVector{3,Float64}(undef)
    db_i_s9 = MVector{3,Float64}(undef)
    dz0_s10 = MVector{3,Float64}(undef)
    dr0inv_s11 = MVector{3,Float64}(undef)
    local dr0invdr_s12
    local rootpq_s13
    local jju_s14
    local jjup_s15
    local jjdu_s16
    local jjdup_s17
    rinv_s18 = (1.0 / r)
    ux_s19 = (x * rinv_s18)
    uy_s20 = (y * rinv_s18)
    uz_s21 = (z * rinv_s18)
    r0inv_s1 = (1.0 / sqrt(((r * r) + (z0 * z0))))
    a_r_s2 = (z0 * r0inv_s1)
    a_i_s3 = ((-z) * r0inv_s1)
    b_r_s4 = (y * r0inv_s1)
    b_i_s5 = ((-x) * r0inv_s1)
    dr0invdr_s12 = ((-(r0inv_s1 ^ 3.0)) * (r + (z0 * dz0dr)))
    dr0inv_s11[(Int32(0))+Int32(1)] = (dr0invdr_s12 * ux_s19)
    dr0inv_s11[(Int32(1))+Int32(1)] = (dr0invdr_s12 * uy_s20)
    dr0inv_s11[(Int32(2))+Int32(1)] = (dr0invdr_s12 * uz_s21)
    dz0_s10[(Int32(0))+Int32(1)] = (dz0dr * ux_s19)
    dz0_s10[(Int32(1))+Int32(1)] = (dz0dr * uy_s20)
    dz0_s10[(Int32(2))+Int32(1)] = (dz0dr * uz_s21)
    k_s22 = Int32(0)
    while (k_s22 < Int32(3))
        da_r_s6[(k_s22)+Int32(1)] = ((dz0_s10[(k_s22)+Int32(1)] * r0inv_s1) + (z0 * dr0inv_s11[(k_s22)+Int32(1)]))
        da_i_s7[(k_s22)+Int32(1)] = ((-z) * dr0inv_s11[(k_s22)+Int32(1)])
        k_s22 += Int32(1)
    end
    da_i_s7[(Int32(2))+Int32(1)] += (-r0inv_s1)
    k_s23 = Int32(0)
    while (k_s23 < Int32(3))
        db_r_s8[(k_s23)+Int32(1)] = (y * dr0inv_s11[(k_s23)+Int32(1)])
        db_i_s9[(k_s23)+Int32(1)] = ((-x) * dr0inv_s11[(k_s23)+Int32(1)])
        k_s23 += Int32(1)
    end
    db_i_s9[(Int32(0))+Int32(1)] += (-r0inv_s1)
    db_r_s8[(Int32(1))+Int32(1)] += r0inv_s1
    k_s24 = Int32(0)
    while (k_s24 < Int32(3))
        dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * Int32(0))) + (((num_atoms * num_nbor) * idxdu_max) * k_s24)))+Int32(1)] = 0.0
        dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * Int32(0))) + (((num_atoms * num_nbor) * idxdu_max) * k_s24)))+Int32(2)] = 0.0
        k_s24 += Int32(1)
    end
    jju_s14 = Int32(1)
    jjdu_s16 = Int32(1)
    j_s25 = Int32(1)
    while (j_s25 <= twojmax)
        deljju_s26 = (j_s25 + Int32(1))
        mb_s27 = Int32(0)
        while ((Int32(2) * mb_s27) <= j_s25)
            k_s28 = Int32(0)
            while (k_s28 < Int32(3))
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * k_s28)))+Int32(1)] = 0.0
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * k_s28)))+Int32(2)] = 0.0
                k_s28 += Int32(1)
            end
            jju_s14 += deljju_s26
            jjdu_s16 += deljju_s26
            mb_s27 += Int32(1)
        end
        ncolhalf_s29 = div(deljju_s26,Int32(2))
        jju_s14 += (deljju_s26 * ncolhalf_s29)
        j_s25 += Int32(1)
    end
    jju_s14 = Int32(1)
    jjdu_s16 = Int32(1)
    jjup_s15 = Int32(0)
    jjdup_s17 = Int32(0)
    j_s30 = Int32(1)
    while (j_s30 <= twojmax)
        deljju_s31 = (j_s30 + Int32(1))
        deljjup_s32 = j_s30
        mb_s33 = Int32(0)
        while ((Int32(2) * mb_s33) < j_s30)
            ma_s34 = Int32(0)
            while (ma_s34 < j_s30)
                up_r_s35 = ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjup_s15)))+Int32(1)]
                up_i_s36 = ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjup_s15)))+Int32(2)]
                rootpq_s13 = rootpqarray[(((j_s30 - ma_s34) + (jdimpq * (j_s30 - mb_s33))))+Int32(1)]
                k_s37 = Int32(0)
                while (k_s37 < Int32(3))
                    dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * k_s37)))+Int32(1)] += (rootpq_s13 * ((((da_r_s6[(k_s37)+Int32(1)] * up_r_s35) + (da_i_s7[(k_s37)+Int32(1)] * up_i_s36)) + (a_r_s2 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s37)))+Int32(1)])) + (a_i_s3 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s37)))+Int32(2)])))
                    dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * k_s37)))+Int32(2)] += (rootpq_s13 * ((((da_r_s6[(k_s37)+Int32(1)] * up_i_s36) - (da_i_s7[(k_s37)+Int32(1)] * up_r_s35)) + (a_r_s2 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s37)))+Int32(2)])) - (a_i_s3 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s37)))+Int32(1)])))
                    k_s37 += Int32(1)
                end
                rootpq_s13 = rootpqarray[(((ma_s34 + Int32(1)) + (jdimpq * (j_s30 - mb_s33))))+Int32(1)]
                k_s38 = Int32(0)
                while (k_s38 < Int32(3))
                    dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * (jjdu_s16 + Int32(1)))) + (((num_atoms * num_nbor) * idxdu_max) * k_s38)))+Int32(1)] = ((-rootpq_s13) * ((((db_r_s8[(k_s38)+Int32(1)] * up_r_s35) + (db_i_s9[(k_s38)+Int32(1)] * up_i_s36)) + (b_r_s4 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s38)))+Int32(1)])) + (b_i_s5 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s38)))+Int32(2)])))
                    dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * (jjdu_s16 + Int32(1)))) + (((num_atoms * num_nbor) * idxdu_max) * k_s38)))+Int32(2)] = ((-rootpq_s13) * ((((db_r_s8[(k_s38)+Int32(1)] * up_i_s36) - (db_i_s9[(k_s38)+Int32(1)] * up_r_s35)) + (b_r_s4 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s38)))+Int32(2)])) - (b_i_s5 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s38)))+Int32(1)])))
                    k_s38 += Int32(1)
                end
                if ((Int32(2) * (mb_s33 + Int32(1))) == j_s30)
                    rootpq_s13 = rootpqarray[(((j_s30 - ma_s34) + (jdimpq * (mb_s33 + Int32(1)))))+Int32(1)]
                    k_s39 = Int32(0)
                    while (k_s39 < Int32(3))
                        dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * (jjdu_s16 + deljju_s31))) + (((num_atoms * num_nbor) * idxdu_max) * k_s39)))+Int32(1)] += (rootpq_s13 * ((((db_r_s8[(k_s39)+Int32(1)] * up_r_s35) - (db_i_s9[(k_s39)+Int32(1)] * up_i_s36)) + (b_r_s4 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s39)))+Int32(1)])) - (b_i_s5 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s39)))+Int32(2)])))
                        dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * (jjdu_s16 + deljju_s31))) + (((num_atoms * num_nbor) * idxdu_max) * k_s39)))+Int32(2)] += (rootpq_s13 * ((((db_r_s8[(k_s39)+Int32(1)] * up_i_s36) + (db_i_s9[(k_s39)+Int32(1)] * up_r_s35)) + (b_r_s4 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s39)))+Int32(2)])) + (b_i_s5 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s39)))+Int32(1)])))
                        k_s39 += Int32(1)
                    end
                    rootpq_s13 = rootpqarray[(((ma_s34 + Int32(1)) + (jdimpq * (mb_s33 + Int32(1)))))+Int32(1)]
                    k_s40 = Int32(0)
                    while (k_s40 < Int32(3))
                        dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * ((jjdu_s16 + Int32(1)) + deljju_s31))) + (((num_atoms * num_nbor) * idxdu_max) * k_s40)))+Int32(1)] = (rootpq_s13 * ((((da_r_s6[(k_s40)+Int32(1)] * up_r_s35) - (da_i_s7[(k_s40)+Int32(1)] * up_i_s36)) + (a_r_s2 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s40)))+Int32(1)])) - (a_i_s3 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s40)))+Int32(2)])))
                        dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * ((jjdu_s16 + Int32(1)) + deljju_s31))) + (((num_atoms * num_nbor) * idxdu_max) * k_s40)))+Int32(2)] = (rootpq_s13 * ((((da_r_s6[(k_s40)+Int32(1)] * up_i_s36) + (da_i_s7[(k_s40)+Int32(1)] * up_r_s35)) + (a_r_s2 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s40)))+Int32(2)])) + (a_i_s3 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdup_s17)) + (((num_atoms * num_nbor) * idxdu_max) * k_s40)))+Int32(1)])))
                        k_s40 += Int32(1)
                    end
                end
                jju_s14 += Int32(1)
                jjup_s15 += Int32(1)
                jjdu_s16 += Int32(1)
                jjdup_s17 += Int32(1)
                ma_s34 += Int32(1)
            end
            jju_s14 += Int32(1)
            jjdu_s16 += Int32(1)
            mb_s33 += Int32(1)
        end
        if ((j_s30 % Int32(2)) == Int32(0))
            jju_s14 += deljju_s31
            jjdu_s16 += deljju_s31
        end
        ncolhalf_s41 = div(deljju_s31,Int32(2))
        jju_s14 += (deljju_s31 * ncolhalf_s41)
        ncolhalfp_s42 = div(deljjup_s32,Int32(2))
        jjup_s15 += (deljjup_s32 * ncolhalfp_s42)
        j_s30 += Int32(1)
    end
    sfac_s43 = compute_sfac(r,rcut,switch_flag)
    dsfac_s44 = compute_dsfac(r,rcut,switch_flag)
    sfac_s43 *= wj_in
    dsfac_s44 *= wj_in
    jju_s14 = Int32(0)
    jjdu_s16 = Int32(0)
    j_s45 = Int32(0)
    while (j_s45 <= twojmax)
        deljju_s46 = (j_s45 + Int32(1))
        mb_s47 = Int32(0)
        while ((Int32(2) * mb_s47) <= j_s45)
            ma_s48 = Int32(0)
            while (ma_s48 <= j_s45)
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(0))))+Int32(1)] = (((dsfac_s44 * ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jju_s14)))+Int32(1)]) * ux_s19) + (sfac_s43 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(0))))+Int32(1)]))
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(0))))+Int32(2)] = (((dsfac_s44 * ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jju_s14)))+Int32(2)]) * ux_s19) + (sfac_s43 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(0))))+Int32(2)]))
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(1))))+Int32(1)] = (((dsfac_s44 * ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jju_s14)))+Int32(1)]) * uy_s20) + (sfac_s43 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(1))))+Int32(1)]))
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(1))))+Int32(2)] = (((dsfac_s44 * ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jju_s14)))+Int32(2)]) * uy_s20) + (sfac_s43 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(1))))+Int32(2)]))
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(2))))+Int32(1)] = (((dsfac_s44 * ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jju_s14)))+Int32(1)]) * uz_s21) + (sfac_s43 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(2))))+Int32(1)]))
                dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(2))))+Int32(2)] = (((dsfac_s44 * ulist[Int32(2)*(((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jju_s14)))+Int32(2)]) * uz_s21) + (sfac_s43 * dulist[Int32(2)*((((natom + (num_atoms * nbor)) + ((num_atoms * num_nbor) * jjdu_s16)) + (((num_atoms * num_nbor) * idxdu_max) * Int32(2))))+Int32(2)]))
                jju_s14 += Int32(1)
                jjdu_s16 += Int32(1)
                ma_s48 += Int32(1)
            end
            mb_s47 += Int32(1)
        end
        ncolhalf_s49 = div(deljju_s46,Int32(2))
        jju_s14 += (deljju_s46 * ncolhalf_s49)
        j_s45 += Int32(1)
    end
    end
    return nothing
end

function reset_ulisttot(ulisttot,ulisttot_size,snap_ix,snap_iy)
    @inbounds begin
    i_s1 = snap_ix
    if (i_s1 < ulisttot_size)
        ulisttot[Int32(2)*(i_s1)+Int32(1)] = 0.0
        ulisttot[Int32(2)*(i_s1)+Int32(2)] = 0.0
    end
    end
    return nothing
end

function set_ulisttot(ulisttot,idxu_block,num_atoms,twojmax,wself,snap_ix,snap_iy)
    @inbounds begin
    natom_s1 = snap_ix
    if (natom_s1 < num_atoms)
        j_s2 = Int32(0)
        while (j_s2 <= twojmax)
            jju_s3 = idxu_block[(j_s2)+Int32(1)]
            ma_s4 = Int32(0)
            while (ma_s4 <= j_s2)
                ulisttot[Int32(2)*((natom_s1 + (num_atoms * jju_s3)))+Int32(1)] = wself
                ulisttot[Int32(2)*((natom_s1 + (num_atoms * jju_s3)))+Int32(2)] = 0.0
                jju_s3 += (j_s2 + Int32(2))
                ma_s4 += Int32(1)
            end
            j_s2 += Int32(1)
        end
    end
    end
    return nothing
end

function update_ulisttot(rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq,snap_ix,snap_iy)
    @inbounds begin
    natom_s1 = snap_ix
    nbor_s2 = snap_iy
    if ((natom_s1 < num_atoms) && (nbor_s2 < num_nbor))
        x_s3 = rij[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(0))))+Int32(1)]
        y_s4 = rij[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(1))))+Int32(1)]
        z_s5 = rij[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(2))))+Int32(1)]
        rsq_s6 = (((x_s3 * x_s3) + (y_s4 * y_s4)) + (z_s5 * z_s5))
        r_s7 = sqrt(rsq_s6)
        theta0_s8 = ((((r_s7 - 0.0) * 0.99363) * 3.14159265358979323846) / (rcutij[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)] - 0.0))
        z0_s9 = (r_s7 / tan(theta0_s8))
        local rootpq_s10
        local jju_s11
        local jjup_s12
        r0inv_s13 = (1.0 / sqrt(((r_s7 * r_s7) + (z0_s9 * z0_s9))))
        a_r_s14 = (r0inv_s13 * z0_s9)
        a_i_s15 = ((-r0inv_s13) * z_s5)
        b_r_s16 = (r0inv_s13 * y_s4)
        b_i_s17 = ((-r0inv_s13) * x_s3)
        local sfac_s18
        sfac_s18 = compute_sfac(r_s7,rcutij[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)],switch_flag)
        sfac_s18 *= wj[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)]
        ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(0))))+Int32(1)] = 1.0
        ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(0))))+Int32(2)] = 0.0
        jju_s11 = Int32(1)
        j_s19 = Int32(1)
        while (j_s19 <= twojmax)
            deljju_s20 = (j_s19 + Int32(1))
            mb_s21 = Int32(0)
            while ((Int32(2) * mb_s21) <= j_s19)
                ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jju_s11)))+Int32(1)] = 0.0
                ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jju_s11)))+Int32(2)] = 0.0
                jju_s11 += deljju_s20
                mb_s21 += Int32(1)
            end
            ncolhalf_s22 = div(deljju_s20,Int32(2))
            jju_s11 += (deljju_s20 * ncolhalf_s22)
            j_s19 += Int32(1)
        end
        jju_s11 = Int32(1)
        jjup_s12 = Int32(0)
        j_s23 = Int32(1)
        while (j_s23 <= twojmax)
            deljju_s24 = (j_s23 + Int32(1))
            deljjup_s25 = j_s23
            mb_max_s26 = div((j_s23 + Int32(1)),Int32(2))
            ma_max_s27 = j_s23
            m_max_s28 = (ma_max_s27 * mb_max_s26)
            m_iter_s29 = Int32(0)
            while (m_iter_s29 < m_max_s28)
                mb_s30 = div(m_iter_s29,ma_max_s27)
                ma_s31 = (m_iter_s29 % ma_max_s27)
                up_r_s32 = ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjup_s12)))+Int32(1)]
                up_i_s33 = ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjup_s12)))+Int32(2)]
                rootpq_s10 = rootpqarray[(((j_s23 - ma_s31) + (jdimpq * (j_s23 - mb_s30))))+Int32(1)]
                ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jju_s11)))+Int32(1)] += (rootpq_s10 * ((a_r_s14 * up_r_s32) + (a_i_s15 * up_i_s33)))
                ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jju_s11)))+Int32(2)] += (rootpq_s10 * ((a_r_s14 * up_i_s33) - (a_i_s15 * up_r_s32)))
                rootpq_s10 = rootpqarray[(((ma_s31 + Int32(1)) + (jdimpq * (j_s23 - mb_s30))))+Int32(1)]
                ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * (jju_s11 + Int32(1)))))+Int32(1)] = ((-rootpq_s10) * ((b_r_s16 * up_r_s32) + (b_i_s17 * up_i_s33)))
                ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * (jju_s11 + Int32(1)))))+Int32(2)] = ((-rootpq_s10) * ((b_r_s16 * up_i_s33) - (b_i_s17 * up_r_s32)))
                if ((Int32(2) * (mb_s30 + Int32(1))) == j_s23)
                    rootpq_s10 = rootpqarray[(((j_s23 - ma_s31) + (jdimpq * (mb_s30 + Int32(1)))))+Int32(1)]
                    ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * (jju_s11 + deljju_s24))))+Int32(1)] += (rootpq_s10 * ((b_r_s16 * up_r_s32) - (b_i_s17 * up_i_s33)))
                    ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * (jju_s11 + deljju_s24))))+Int32(2)] += (rootpq_s10 * ((b_r_s16 * up_i_s33) + (b_i_s17 * up_r_s32)))
                    rootpq_s10 = rootpqarray[(((ma_s31 + Int32(1)) + (jdimpq * (mb_s30 + Int32(1)))))+Int32(1)]
                    ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * ((jju_s11 + deljju_s24) + Int32(1)))))+Int32(1)] = (rootpq_s10 * ((a_r_s14 * up_r_s32) - (a_i_s15 * up_i_s33)))
                    ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * ((jju_s11 + deljju_s24) + Int32(1)))))+Int32(2)] = (rootpq_s10 * ((a_r_s14 * up_i_s33) + (a_i_s15 * up_r_s32)))
                end
                jju_s11 += Int32(1)
                jjup_s12 += Int32(1)
                if (ma_s31 == (ma_max_s27 - Int32(1)))
                    jju_s11 += Int32(1)
                end
                m_iter_s29 += Int32(1)
            end
            jjui_s34 = idxu_block[(j_s23)+Int32(1)]
            jjuip_s35 = ((jjui_s34 + ((j_s23 + Int32(1)) * (j_s23 + Int32(1)))) - Int32(1))
            mb_s36 = Int32(0)
            while ((Int32(2) * mb_s36) < j_s23)
                ma_s37 = Int32(0)
                while (ma_s37 <= j_s23)
                    ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjuip_s35)))+Int32(1)] = (ulist_parity[(jjui_s34)+Int32(1)] * ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjui_s34)))+Int32(1)])
                    ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjuip_s35)))+Int32(2)] = (ulist_parity[(jjui_s34)+Int32(1)] * (-ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjui_s34)))+Int32(2)]))
                    jjui_s34 += Int32(1)
                    jjuip_s35 -= Int32(1)
                    ma_s37 += Int32(1)
                end
                mb_s36 += Int32(1)
            end
            if ((j_s23 % Int32(2)) == Int32(0))
                jju_s11 += deljju_s24
            end
            ncolhalf_s38 = div(deljju_s24,Int32(2))
            jju_s11 += (deljju_s24 * ncolhalf_s38)
            ncolhalfp_s39 = div(deljjup_s25,Int32(2))
            jjup_s12 += (deljjup_s25 * ncolhalfp_s39)
            j_s23 += Int32(1)
        end
        sfac_s18 = compute_sfac(r_s7,rcutij[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)],switch_flag)
        sfac_s18 *= wj[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)]
        j_s40 = Int32(0)
        while (j_s40 <= twojmax)
            jju_s41 = idxu_block[(j_s40)+Int32(1)]
            mb_s42 = Int32(0)
            while (mb_s42 <= j_s40)
                ma_s43 = Int32(0)
                while (ma_s43 <= j_s40)
                    snap_atomic!(ulisttot,Int32(2)*((natom_s1 + (num_atoms * jju_s41)))+Int32(1),(sfac_s18 * ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jju_s41)))+Int32(1)]))
                    snap_atomic!(ulisttot,Int32(2)*((natom_s1 + (num_atoms * jju_s41)))+Int32(2),(sfac_s18 * ulist[Int32(2)*(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jju_s41)))+Int32(2)]))
                    jju_s41 += Int32(1)
                    ma_s43 += Int32(1)
                end
                mb_s42 += Int32(1)
            end
            j_s40 += Int32(1)
        end
    end
    end
    return nothing
end

function reset_ylist(ylist,ylist_size,snap_ix,snap_iy)
    @inbounds begin
    i_s1 = snap_ix
    if (i_s1 < ylist_size)
        ylist[Int32(2)*(i_s1)+Int32(1)] = 0.0
        ylist[Int32(2)*(i_s1)+Int32(2)] = 0.0
    end
    end
    return nothing
end

function compute_yi(idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim,snap_ix,snap_iy)
    @inbounds begin
    natom_s1 = snap_ix
    jjz_s2 = snap_iy
    if ((jjz_s2 < idxz_max) && (natom_s1 < num_atoms))
        j1_s3 = idxz[((jjz_s2 + (idxz_max * Int32(0))))+Int32(1)]
        j2_s4 = idxz[((jjz_s2 + (idxz_max * Int32(1))))+Int32(1)]
        j_s5 = idxz[((jjz_s2 + (idxz_max * Int32(2))))+Int32(1)]
        ma1min_s6 = idxz[((jjz_s2 + (idxz_max * Int32(3))))+Int32(1)]
        ma2max_s7 = idxz[((jjz_s2 + (idxz_max * Int32(4))))+Int32(1)]
        na_s8 = idxz[((jjz_s2 + (idxz_max * Int32(5))))+Int32(1)]
        mb1min_s9 = idxz[((jjz_s2 + (idxz_max * Int32(6))))+Int32(1)]
        mb2max_s10 = idxz[((jjz_s2 + (idxz_max * Int32(7))))+Int32(1)]
        nb_s11 = idxz[((jjz_s2 + (idxz_max * Int32(8))))+Int32(1)]
        betaj_s12 = idxzbeta[(jjz_s2)+Int32(1)]
        cgblock_s13 = SnapShift(cglist,idxcg_block[(((j1_s3 + (jdim * j2_s4)) + ((jdim * jdim) * j_s5)))+Int32(1)])
        mb_s14 = div(((((Int32(2) * (mb1min_s9 + mb2max_s10)) - j1_s3) - j2_s4) + j_s5),Int32(2))
        ma_s15 = div(((((Int32(2) * (ma1min_s6 + ma2max_s7)) - j1_s3) - j2_s4) + j_s5),Int32(2))
        jjdu_s16 = ((idxdu_block[(j_s5)+Int32(1)] + ((j_s5 + Int32(1)) * mb_s14)) + ma_s15)
        jju1_s17 = (idxu_block[(j1_s3)+Int32(1)] + ((j1_s3 + Int32(1)) * mb1min_s9))
        jju2_s18 = (idxu_block[(j2_s4)+Int32(1)] + ((j2_s4 + Int32(1)) * mb2max_s10))
        icgb_s19 = ((mb1min_s9 * (j2_s4 + Int32(1))) + mb2max_s10)
        ztmp_r_s20 = 0.0
        ztmp_i_s21 = 0.0
        ib_s22 = Int32(0)
        while (ib_s22 < nb_s11)
            suma1_r_s23 = 0.0
            suma1_i_s24 = 0.0
            ma1_s25 = ma1min_s6
            ma2_s26 = ma2max_s7
            icga_s27 = ((ma1min_s6 * (j2_s4 + Int32(1))) + ma2max_s7)
            ia_s28 = Int32(0)
            while (ia_s28 < na_s8)
                suma1_r_s23 += (cgblock_s13[(icga_s27)+Int32(1)] * ((ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju1_s17 + ma1_s25))))+Int32(1)] * ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju2_s18 + ma2_s26))))+Int32(1)]) - (ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju1_s17 + ma1_s25))))+Int32(2)] * ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju2_s18 + ma2_s26))))+Int32(2)])))
                suma1_i_s24 += (cgblock_s13[(icga_s27)+Int32(1)] * ((ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju1_s17 + ma1_s25))))+Int32(1)] * ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju2_s18 + ma2_s26))))+Int32(2)]) + (ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju1_s17 + ma1_s25))))+Int32(2)] * ulisttot[Int32(2)*((natom_s1 + (num_atoms * (jju2_s18 + ma2_s26))))+Int32(1)])))
                ma1_s25 += Int32(1)
                ma2_s26 -= Int32(1)
                icga_s27 += j2_s4
                ia_s28 += Int32(1)
            end
            ztmp_r_s20 += (cgblock_s13[(icgb_s19)+Int32(1)] * suma1_r_s23)
            ztmp_i_s21 += (cgblock_s13[(icgb_s19)+Int32(1)] * suma1_i_s24)
            jju1_s17 += (j1_s3 + Int32(1))
            jju2_s18 -= (j2_s4 + Int32(1))
            icgb_s19 += j2_s4
            ib_s22 += Int32(1)
        end
        snap_atomic!(ylist,Int32(2)*((natom_s1 + (num_atoms * jjdu_s16)))+Int32(1),(betaj_s12 * ztmp_r_s20))
        snap_atomic!(ylist,Int32(2)*((natom_s1 + (num_atoms * jjdu_s16)))+Int32(2),(betaj_s12 * ztmp_i_s21))
    end
    end
    return nothing
end

function compute_duidrj(wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag,snap_ix,snap_iy)
    @inbounds begin
    natom_s1 = snap_ix
    nbor_s2 = snap_iy
    if ((natom_s1 < num_atoms) && (nbor_s2 < num_nbor))
        wj_in_s3 = wj[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)]
        rcut_s4 = rcutij[((natom_s1 + (num_atoms * nbor_s2)))+Int32(1)]
        x_s5 = rij[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(0))))+Int32(1)]
        y_s6 = rij[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(1))))+Int32(1)]
        z_s7 = rij[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * Int32(2))))+Int32(1)]
        rsq_s8 = (((x_s5 * x_s5) + (y_s6 * y_s6)) + (z_s7 * z_s7))
        r_s9 = sqrt(rsq_s8)
        rscale0_s10 = ((0.99363 * 3.14159265358979323846) / (rcut_s4 - 0.0))
        theta0_s11 = ((r_s9 - 0.0) * rscale0_s10)
        cs_s12 = cos(theta0_s11)
        sn_s13 = sin(theta0_s11)
        z0_s14 = ((r_s9 * cs_s12) / sn_s13)
        dz0dr_s15 = ((z0_s14 / r_s9) - (((r_s9 * rscale0_s10) * (rsq_s8 + (z0_s14 * z0_s14))) / rsq_s8))
        compute_duarray(natom_s1,nbor_s2,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag,x_s5,y_s6,z_s7,z0_s14,r_s9,dz0dr_s15,wj_in_s3,rcut_s4,rootpqarray,ulist,dulist)
    end
    end
    return nothing
end

function compute_deidrj(idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max,snap_ix,snap_iy)
    @inbounds begin
    natom_s1 = snap_ix
    nbor_s2 = snap_iy
    if ((natom_s1 < num_atoms) && (nbor_s2 < num_nbor))
        k_s3 = Int32(0)
        while (k_s3 < Int32(3))
            dedr[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * k_s3)))+Int32(1)] = 0.0
            k_s3 += Int32(1)
        end
        j_s4 = Int32(0)
        while (j_s4 <= twojmax)
            jjdu_s5 = idxdu_block[(j_s4)+Int32(1)]
            mb_s6 = Int32(0)
            while ((Int32(2) * mb_s6) < j_s4)
                ma_s7 = Int32(0)
                while (ma_s7 <= j_s4)
                    jjjmambyarray_r_s8 = ylist[Int32(2)*((natom_s1 + (num_atoms * jjdu_s5)))+Int32(1)]
                    jjjmambyarray_i_s9 = ylist[Int32(2)*((natom_s1 + (num_atoms * jjdu_s5)))+Int32(2)]
                    k_s10 = Int32(0)
                    while (k_s10 < Int32(3))
                        dedr[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * k_s10)))+Int32(1)] += ((dulist[Int32(2)*((((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjdu_s5)) + (((num_atoms * num_nbor) * idxdu_max) * k_s10)))+Int32(1)] * jjjmambyarray_r_s8) + (dulist[Int32(2)*((((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjdu_s5)) + (((num_atoms * num_nbor) * idxdu_max) * k_s10)))+Int32(2)] * jjjmambyarray_i_s9))
                        k_s10 += Int32(1)
                    end
                    jjdu_s5 += Int32(1)
                    ma_s7 += Int32(1)
                end
                mb_s6 += Int32(1)
            end
            if ((j_s4 % Int32(2)) == Int32(0))
                mb_s11 = div(j_s4,Int32(2))
                ma_s12 = Int32(0)
                while (ma_s12 < mb_s11)
                    jjjmambyarray_r_s13 = ylist[Int32(2)*((natom_s1 + (num_atoms * jjdu_s5)))+Int32(1)]
                    jjjmambyarray_i_s14 = ylist[Int32(2)*((natom_s1 + (num_atoms * jjdu_s5)))+Int32(2)]
                    k_s15 = Int32(0)
                    while (k_s15 < Int32(3))
                        dedr[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * k_s15)))+Int32(1)] += ((dulist[Int32(2)*((((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjdu_s5)) + (((num_atoms * num_nbor) * idxdu_max) * k_s15)))+Int32(1)] * jjjmambyarray_r_s13) + (dulist[Int32(2)*((((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjdu_s5)) + (((num_atoms * num_nbor) * idxdu_max) * k_s15)))+Int32(2)] * jjjmambyarray_i_s14))
                        k_s15 += Int32(1)
                    end
                    jjdu_s5 += Int32(1)
                    ma_s12 += Int32(1)
                end
                jjjmambyarray_r_s16 = ylist[Int32(2)*((natom_s1 + (num_atoms * jjdu_s5)))+Int32(1)]
                jjjmambyarray_i_s17 = ylist[Int32(2)*((natom_s1 + (num_atoms * jjdu_s5)))+Int32(2)]
                k_s18 = Int32(0)
                while (k_s18 < Int32(3))
                    dedr[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * k_s18)))+Int32(1)] += (((dulist[Int32(2)*((((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjdu_s5)) + (((num_atoms * num_nbor) * idxdu_max) * k_s18)))+Int32(1)] * jjjmambyarray_r_s16) + (dulist[Int32(2)*((((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * jjdu_s5)) + (((num_atoms * num_nbor) * idxdu_max) * k_s18)))+Int32(2)] * jjjmambyarray_i_s17)) * 0.5)
                    k_s18 += Int32(1)
                end
                jjdu_s5 += Int32(1)
            end
            j_s4 += Int32(1)
        end
        k_s19 = Int32(0)
        while (k_s19 < Int32(3))
            dedr[(((natom_s1 + (num_atoms * nbor_s2)) + ((num_atoms * num_nbor) * k_s19)))+Int32(1)] *= 2.0
            k_s19 += Int32(1)
        end
    end
    end
    return nothing
end
