using Atomix
@inline function cmul(ar::Float64, ai::Float64, br::Float64, bi::Float64)
    return ar * br - ai * bi, ar * bi + ai * br
end

@inline function gpp_thread!(tx,bx,by,gx,gy,
    number_bands::Int32, ngpown::Int32, ncouls::Int32,
    inv_igp_index, indinv, wx_array,
    wtilde_re, wtilde_im, aqsm_re, aqsm_im, aqsn_re, aqsn_im,
    ieps_re, ieps_im, vcoul, achtemp_re, achtemp_im)

    local_re0 = 0.0
    local_im0 = 0.0
    local_re1 = 0.0
    local_im1 = 0.0
    local_re2 = 0.0
    local_im2 = 0.0

    n1 = bx
    while n1 < number_bands
        my_igp = by
        while my_igp < ngpown
            indigp = inv_igp_index[my_igp + Int32(1)]
            igp = indinv[indigp + Int32(1)]

            aqsm_idx = n1 * ncouls + igp + Int32(1)
            sch_r, sch_i = cmul(aqsm_re[aqsm_idx], -aqsm_im[aqsm_idx],
                                aqsn_re[aqsm_idx], aqsn_im[aqsm_idx])
            scale = 0.5 * vcoul[igp + Int32(1)]
            sch_r *= scale
            sch_i *= scale

            ig = tx
            while ig < ncouls
                eps_idx = my_igp * ncouls + ig + Int32(1)
                wt_r = wtilde_re[eps_idx]
                wt_i = wtilde_im[eps_idx]
                ie_r = ieps_re[eps_idx]
                ie_i = ieps_im[eps_idx]

                w = wx_array[1]
                wd_r = w - wt_r
                wd_i = -wt_i
                denom = wd_r * wd_r + wd_i * wd_i
                t_r, t_i = cmul(wt_r, wt_i, wd_r, -wd_i)
                del_r = t_r / denom
                del_i = t_i / denom
                arr_r, arr_i = cmul(del_r, del_i, ie_r, ie_i)
                arr_r, arr_i = cmul(arr_r, arr_i, sch_r, sch_i)
                local_re0 += arr_r
                local_im0 += arr_i

                w = wx_array[2]
                wd_r = w - wt_r
                wd_i = -wt_i
                denom = wd_r * wd_r + wd_i * wd_i
                t_r, t_i = cmul(wt_r, wt_i, wd_r, -wd_i)
                del_r = t_r / denom
                del_i = t_i / denom
                arr_r, arr_i = cmul(del_r, del_i, ie_r, ie_i)
                arr_r, arr_i = cmul(arr_r, arr_i, sch_r, sch_i)
                local_re1 += arr_r
                local_im1 += arr_i

                w = wx_array[3]
                wd_r = w - wt_r
                wd_i = -wt_i
                denom = wd_r * wd_r + wd_i * wd_i
                t_r, t_i = cmul(wt_r, wt_i, wd_r, -wd_i)
                del_r = t_r / denom
                del_i = t_i / denom
                arr_r, arr_i = cmul(del_r, del_i, ie_r, ie_i)
                arr_r, arr_i = cmul(arr_r, arr_i, sch_r, sch_i)
                local_re2 += arr_r
                local_im2 += arr_i

                ig += Int32(32)
            end
            my_igp += gy
        end
        n1 += gx
    end

    Atomix.@atomic achtemp_re[1] += local_re0
    Atomix.@atomic achtemp_im[1] += local_im0
    Atomix.@atomic achtemp_re[2] += local_re1
    Atomix.@atomic achtemp_im[2] += local_im1
    Atomix.@atomic achtemp_re[3] += local_re2
    Atomix.@atomic achtemp_im[3] += local_im2
    return
end
