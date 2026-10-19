# Generated from the unchanged HEARTWALL C kernel AST; see translation_provenance.json.
struct ZeroView{A}
    data::A
    offset::Int32
end
@inline Base.getindex(v::ZeroView, i::Integer) = @inbounds v.data[v.offset+i+Int32(1)]
@inline Base.setindex!(v::ZeroView, x, i::Integer) = (@inbounds v.data[v.offset+i+Int32(1)]=x)
@inline shiftview(v::ZeroView,i::Integer)=ZeroView(v.data,v.offset+Int32(i))
function hw(frame_no::Int32,d_common,buffers)
    bx0=Int32(CUDA.blockIdx().x-1);tx0=Int32(CUDA.threadIdx().x-1)
    d_frame = ZeroView(buffers.frame,Int32(16))
    d_endoRow = ZeroView(buffers.endoRow,Int32(16))
    d_endoCol = ZeroView(buffers.endoCol,Int32(16))
    d_tEndoRowLoc = ZeroView(buffers.tEndoRowLoc,Int32(16))
    d_tEndoColLoc = ZeroView(buffers.tEndoColLoc,Int32(16))
    d_epiRow = ZeroView(buffers.epiRow,Int32(16))
    d_epiCol = ZeroView(buffers.epiCol,Int32(16))
    d_tEpiRowLoc = ZeroView(buffers.tEpiRowLoc,Int32(16))
    d_tEpiColLoc = ZeroView(buffers.tEpiColLoc,Int32(16))
    d_endoT = ZeroView(buffers.endoT,Int32(16))
    d_epiT = ZeroView(buffers.epiT,Int32(16))
    d_in2 = ZeroView(buffers.in2,Int32(16))
    d_conv = ZeroView(buffers.conv,Int32(16))
    d_in2_pad_cumv = ZeroView(buffers.in2_pad_cumv,Int32(16))
    d_in2_pad_cumv_sel = ZeroView(buffers.in2_pad_cumv_sel,Int32(16))
    d_in2_sub_cumh = ZeroView(buffers.in2_sub_cumh,Int32(16))
    d_in2_sub_cumh_sel = ZeroView(buffers.in2_sub_cumh_sel,Int32(16))
    d_in2_sub2 = ZeroView(buffers.in2_sub2,Int32(16))
    d_in2_sqr = ZeroView(buffers.in2_sqr,Int32(16))
    d_in2_sqr_sub2 = ZeroView(buffers.in2_sqr_sub2,Int32(16))
    d_in_sqr = ZeroView(buffers.in_sqr,Int32(16))
    d_tMask = ZeroView(buffers.tMask,Int32(16))
    d_mask_conv = ZeroView(buffers.mask_conv,Int32(16))
    d_in_mod_temp = ZeroView(buffers.in_mod_temp,Int32(16))
    d_in_partial_sum = ZeroView(buffers.in_partial_sum,Int32(16))
    d_in_sqr_partial_sum = ZeroView(buffers.in_sqr_partial_sum,Int32(16))
    d_par_max_val = ZeroView(buffers.par_max_val,Int32(16))
    d_par_max_coo = ZeroView(buffers.par_max_coo,Int32(16))
    d_in_final_sum = ZeroView(buffers.in_final_sum,Int32(16))
    d_in_sqr_final_sum = ZeroView(buffers.in_sqr_final_sum,Int32(16))
    d_denomT = ZeroView(buffers.denomT,Int32(16))
    local rot_row::Int32
    local rot_col::Int32
    local in2_rowlow::Int32
    local in2_collow::Int32
    local ic::Int32
    local jc::Int32
    local jp1::Int32
    local ja1::Int32
    local ja2::Int32
    local ip1::Int32
    local ia1::Int32
    local ia2::Int32
    local ja::Int32
    local jb::Int32
    local ia::Int32
    local ib::Int32
    local s::Float32
    local i::Int32
    local j::Int32
    local row::Int32
    local col::Int32
    local ori_row::Int32
    local ori_col::Int32
    local position::Int32
    local sum::Float32
    local pos_ori::Int32
    local temp::Float32
    local temp2::Float32
    local location::Int32
    local cent::Int32
    local tMask_row::Int32
    local tMask_col::Int32
    local largest_value_current::Float32 = Int32(0)
    local largest_value::Float32 = Int32(0)
    local largest_coordinate_current::Int32 = Int32(0)
    local largest_coordinate::Int32 = Int32(0)
    local fin_max_val::Float32 = Int32(0)
    local fin_max_coo::Int32 = Int32(0)
    local largest_row::Int32
    local largest_col::Int32
    local offset_row::Int32
    local offset_col::Int32
    local mean::Float32
    local mean_sqr::Float32
    local variance::Float32
    local deviation::Float32
    local pointer::Int32
    local ori_pointer::Int32
    local loc_pointer::Int32
    local bx::Int32 = bx0
    local tx::Int32 = tx0
    local ei_new::Int32
    local d_unique_point_no::Int32 = ((bx < d_common.endoPoints) ? bx : (bx - d_common.endoPoints))
    d_unique_d_Row = ((bx < d_common.endoPoints) ? d_endoRow : d_epiRow)
    d_unique_d_Col = ((bx < d_common.endoPoints) ? d_endoCol : d_epiCol)
    d_unique_d_tRowLoc = ((bx < d_common.endoPoints) ? d_tEndoRowLoc : d_tEpiRowLoc)
    d_unique_d_tColLoc = ((bx < d_common.endoPoints) ? d_tEndoColLoc : d_tEpiColLoc)
    d_in = ((bx < d_common.endoPoints) ? shiftview(d_endoT,(d_unique_point_no * d_common.in_elem)) : shiftview(d_epiT,(d_unique_point_no * d_common.in_elem)))
    d_unique_d_in2 = shiftview(d_in2,(bx * d_common.in2_elem))
    d_unique_d_conv = shiftview(d_conv,(bx * d_common.conv_elem))
    d_unique_d_in2_pad_cumv = shiftview(d_in2_pad_cumv,(bx * d_common.in2_pad_cumv_elem))
    d_unique_d_in2_pad_cumv_sel = shiftview(d_in2_pad_cumv_sel,(bx * d_common.in2_pad_cumv_sel_elem))
    d_unique_d_in2_sub_cumh = shiftview(d_in2_sub_cumh,(bx * d_common.in2_sub_cumh_elem))
    d_unique_d_in2_sub_cumh_sel = shiftview(d_in2_sub_cumh_sel,(bx * d_common.in2_sub_cumh_sel_elem))
    d_unique_d_in2_sub2 = shiftview(d_in2_sub2,(bx * d_common.in2_sub2_elem))
    d_unique_d_in2_sqr = shiftview(d_in2_sqr,(bx * d_common.in2_sqr_elem))
    d_unique_d_in2_sqr_sub2 = shiftview(d_in2_sqr_sub2,(bx * d_common.in2_sqr_sub2_elem))
    d_unique_d_in_sqr = shiftview(d_in_sqr,(bx * d_common.in_sqr_elem))
    d_unique_d_tMask = shiftview(d_tMask,(bx * d_common.tMask_elem))
    d_unique_d_mask_conv = shiftview(d_mask_conv,(bx * d_common.mask_conv_elem))
    d_unique_d_in_mod_temp = shiftview(d_in_mod_temp,(bx * d_common.in_elem))
    d_unique_d_in_partial_sum = shiftview(d_in_partial_sum,(bx * d_common.in_cols))
    d_unique_d_in_sqr_partial_sum = shiftview(d_in_sqr_partial_sum,(bx * d_common.in_sqr_rows))
    d_unique_d_par_max_val = shiftview(d_par_max_val,(bx * d_common.mask_conv_rows))
    d_unique_d_par_max_coo = shiftview(d_par_max_coo,(bx * d_common.mask_conv_rows))
    d_unique_d_in_final_sum = shiftview(d_in_final_sum,bx)
    d_unique_d_in_sqr_final_sum = shiftview(d_in_sqr_final_sum,bx)
    d_unique_d_denomT = shiftview(d_denomT,bx)
    if (frame_no == Int32(0))
        ei_new = tx
        if (ei_new == Int32(0))
            pointer = ((d_unique_point_no * d_common.no_frames) + frame_no)
            d_unique_d_tRowLoc[pointer] = d_unique_d_Row[d_unique_point_no]
            d_unique_d_tColLoc[pointer] = d_unique_d_Col[d_unique_point_no]
        end
        ei_new = tx
        while (ei_new < d_common.in_elem)
            row = (((ei_new + Int32(1)) % d_common.in_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in_rows) == Int32(0))
                row = (d_common.in_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = (((d_unique_d_Row[d_unique_point_no] - Int32(25)) + row) - Int32(1))
            ori_col = (((d_unique_d_Col[d_unique_point_no] - Int32(25)) + col) - Int32(1))
            ori_pointer = ((ori_col * d_common.frame_rows) + ori_row)
            d_in[((col * d_common.in_rows) + row)] = d_frame[ori_pointer]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
    end
    if (frame_no != Int32(0))
        in2_rowlow = (d_unique_d_Row[d_unique_point_no] - d_common.sSize)
        in2_collow = (d_unique_d_Col[d_unique_point_no] - d_common.sSize)
        ei_new = tx
        while (ei_new < d_common.in2_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_rows) == Int32(0))
                row = (d_common.in2_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + in2_rowlow) - Int32(1))
            ori_col = ((col + in2_collow) - Int32(1))
            d_unique_d_in2[ei_new] = d_frame[((ori_col * d_common.frame_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in_elem)
            row = (((ei_new + Int32(1)) % d_common.in_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in_rows) == Int32(0))
                row = (d_common.in_rows - Int32(1))
                col = (col - Int32(1))
            end
            rot_row = ((d_common.in_rows - Int32(1)) - row)
            rot_col = ((d_common.in_rows - Int32(1)) - col)
            d_unique_d_in_mod_temp[ei_new] = d_in[((rot_col * d_common.in_rows) + rot_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.conv_elem)
            ic = ((ei_new + Int32(1)) % d_common.conv_rows)
            jc = (div((ei_new + Int32(1)),d_common.conv_rows) + Int32(1))
            if (((ei_new + Int32(1)) % d_common.conv_rows) == Int32(0))
                ic = d_common.conv_rows
                jc = (jc - Int32(1))
            end
            j = (jc + d_common.joffset)
            jp1 = (j + Int32(1))
            if (d_common.in2_cols < jp1)
                ja1 = (jp1 - d_common.in2_cols)
            else
                ja1 = Int32(1)
            end
            if (d_common.in_cols < j)
                ja2 = d_common.in_cols
            else
                ja2 = j
            end
            i = (ic + d_common.ioffset)
            ip1 = (i + Int32(1))
            if (d_common.in2_rows < ip1)
                ia1 = (ip1 - d_common.in2_rows)
            else
                ia1 = Int32(1)
            end
            if (d_common.in_rows < i)
                ia2 = d_common.in_rows
            else
                ia2 = i
            end
            s = Int32(0)
            ja = ja1
            while (ja <= ja2)
                jb = (jp1 - ja)
                ia = ia1
                while (ia <= ia2)
                    ib = (ip1 - ia)
                    s = fma(Float32(d_unique_d_in_mod_temp[(((d_common.in_rows * (ja - Int32(1))) + ia) - Int32(1))]),Float32(d_unique_d_in2[(((d_common.in2_rows * (jb - Int32(1))) + ib) - Int32(1))]),Float32(s))
                    ia += Int32(1)
                end
                ja += Int32(1)
            end
            d_unique_d_conv[ei_new] = s
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_pad_cumv_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_pad_cumv_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_pad_cumv_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_pad_cumv_rows) == Int32(0))
                row = (d_common.in2_pad_cumv_rows - Int32(1))
                col = (col - Int32(1))
            end
            if ((((row > (d_common.in2_pad_add_rows - Int32(1))) && (row < (d_common.in2_pad_add_rows + d_common.in2_rows))) && (col > (d_common.in2_pad_add_cols - Int32(1)))) && (col < (d_common.in2_pad_add_cols + d_common.in2_cols)))
                ori_row = (row - d_common.in2_pad_add_rows)
                ori_col = (col - d_common.in2_pad_add_cols)
                d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2[((ori_col * d_common.in2_rows) + ori_row)]
            else
                d_unique_d_in2_pad_cumv[ei_new] = Int32(0)
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_pad_cumv_cols)
            pos_ori = (ei_new * d_common.in2_pad_cumv_rows)
            sum = Int32(0)
            position = pos_ori
            while (position < (pos_ori + d_common.in2_pad_cumv_rows))
                d_unique_d_in2_pad_cumv[position] = (d_unique_d_in2_pad_cumv[position] + sum)
                sum = d_unique_d_in2_pad_cumv[position]
                position = (position + Int32(1))
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_pad_cumv_sel_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_pad_cumv_sel_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_pad_cumv_sel_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_pad_cumv_sel_rows) == Int32(0))
                row = (d_common.in2_pad_cumv_sel_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_pad_cumv_sel_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_pad_cumv_sel_collow) - Int32(1))
            d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[((ori_col * d_common.in2_pad_cumv_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_sub_cumh_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_sub_cumh_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_sub_cumh_rows) == Int32(0))
                row = (d_common.in2_sub_cumh_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_pad_cumv_sel2_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_pad_cumv_sel2_collow) - Int32(1))
            d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[((ori_col * d_common.in2_pad_cumv_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_elem)
            d_unique_d_in2_sub_cumh[ei_new] = (d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new])
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_rows)
            pos_ori = ei_new
            sum = Int32(0)
            position = pos_ori
            while (position < (pos_ori + d_common.in2_sub_cumh_elem))
                d_unique_d_in2_sub_cumh[position] = (d_unique_d_in2_sub_cumh[position] + sum)
                sum = d_unique_d_in2_sub_cumh[position]
                position = (position + d_common.in2_sub_cumh_rows)
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_sel_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_sub_cumh_sel_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_sub_cumh_sel_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_sub_cumh_sel_rows) == Int32(0))
                row = (d_common.in2_sub_cumh_sel_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_sub_cumh_sel_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_sub_cumh_sel_collow) - Int32(1))
            d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[((ori_col * d_common.in2_sub_cumh_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_sub2_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_sub2_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_sub2_rows) == Int32(0))
                row = (d_common.in2_sub2_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_sub_cumh_sel2_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_sub_cumh_sel2_collow) - Int32(1))
            d_unique_d_in2_sub2[ei_new] = d_unique_d_in2_sub_cumh[((ori_col * d_common.in2_sub_cumh_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            d_unique_d_in2_sub2[ei_new] = (d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sub2[ei_new])
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sqr_elem)
            temp = d_unique_d_in2[ei_new]
            d_unique_d_in2_sqr[ei_new] = (temp * temp)
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_pad_cumv_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_pad_cumv_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_pad_cumv_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_pad_cumv_rows) == Int32(0))
                row = (d_common.in2_pad_cumv_rows - Int32(1))
                col = (col - Int32(1))
            end
            if ((((row > (d_common.in2_pad_add_rows - Int32(1))) && (row < (d_common.in2_pad_add_rows + d_common.in2_sqr_rows))) && (col > (d_common.in2_pad_add_cols - Int32(1)))) && (col < (d_common.in2_pad_add_cols + d_common.in2_sqr_cols)))
                ori_row = (row - d_common.in2_pad_add_rows)
                ori_col = (col - d_common.in2_pad_add_cols)
                d_unique_d_in2_pad_cumv[ei_new] = d_unique_d_in2_sqr[((ori_col * d_common.in2_sqr_rows) + ori_row)]
            else
                d_unique_d_in2_pad_cumv[ei_new] = Int32(0)
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_pad_cumv_cols)
            pos_ori = (ei_new * d_common.in2_pad_cumv_rows)
            sum = Int32(0)
            position = pos_ori
            while (position < (pos_ori + d_common.in2_pad_cumv_rows))
                d_unique_d_in2_pad_cumv[position] = (d_unique_d_in2_pad_cumv[position] + sum)
                sum = d_unique_d_in2_pad_cumv[position]
                position = (position + Int32(1))
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_pad_cumv_sel_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_pad_cumv_sel_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_pad_cumv_sel_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_pad_cumv_sel_rows) == Int32(0))
                row = (d_common.in2_pad_cumv_sel_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_pad_cumv_sel_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_pad_cumv_sel_collow) - Int32(1))
            d_unique_d_in2_pad_cumv_sel[ei_new] = d_unique_d_in2_pad_cumv[((ori_col * d_common.in2_pad_cumv_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_sub_cumh_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_sub_cumh_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_sub_cumh_rows) == Int32(0))
                row = (d_common.in2_sub_cumh_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_pad_cumv_sel2_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_pad_cumv_sel2_collow) - Int32(1))
            d_unique_d_in2_sub_cumh[ei_new] = d_unique_d_in2_pad_cumv[((ori_col * d_common.in2_pad_cumv_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_elem)
            d_unique_d_in2_sub_cumh[ei_new] = (d_unique_d_in2_pad_cumv_sel[ei_new] - d_unique_d_in2_sub_cumh[ei_new])
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_rows)
            pos_ori = ei_new
            sum = Int32(0)
            position = pos_ori
            while (position < (pos_ori + d_common.in2_sub_cumh_elem))
                d_unique_d_in2_sub_cumh[position] = (d_unique_d_in2_sub_cumh[position] + sum)
                sum = d_unique_d_in2_sub_cumh[position]
                position = (position + d_common.in2_sub_cumh_rows)
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub_cumh_sel_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_sub_cumh_sel_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_sub_cumh_sel_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_sub_cumh_sel_rows) == Int32(0))
                row = (d_common.in2_sub_cumh_sel_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_sub_cumh_sel_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_sub_cumh_sel_collow) - Int32(1))
            d_unique_d_in2_sub_cumh_sel[ei_new] = d_unique_d_in2_sub_cumh[((ori_col * d_common.in2_sub_cumh_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            row = (((ei_new + Int32(1)) % d_common.in2_sub2_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in2_sub2_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in2_sub2_rows) == Int32(0))
                row = (d_common.in2_sub2_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = ((row + d_common.in2_sub_cumh_sel2_rowlow) - Int32(1))
            ori_col = ((col + d_common.in2_sub_cumh_sel2_collow) - Int32(1))
            d_unique_d_in2_sqr_sub2[ei_new] = d_unique_d_in2_sub_cumh[((ori_col * d_common.in2_sub_cumh_rows) + ori_row)]
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            d_unique_d_in2_sqr_sub2[ei_new] = (d_unique_d_in2_sub_cumh_sel[ei_new] - d_unique_d_in2_sqr_sub2[ei_new])
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            temp = d_unique_d_in2_sub2[ei_new]
            temp2 = (d_unique_d_in2_sqr_sub2[ei_new] - ((temp * temp) / d_common.in_elem))
            if (temp2 < Int32(0))
                temp2 = Int32(0)
            end
            d_unique_d_in2_sqr_sub2[ei_new] = sqrt(temp2)
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in_sqr_elem)
            temp = d_in[ei_new]
            d_unique_d_in_sqr[ei_new] = (temp * temp)
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in_cols)
            sum = Int32(0)
            i = Int32(0)
            while (i < d_common.in_rows)
                sum = (sum + d_in[((ei_new * d_common.in_rows) + i)])
                i += Int32(1)
            end
            d_unique_d_in_partial_sum[ei_new] = sum
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in_sqr_rows)
            sum = Int32(0)
            i = Int32(0)
            while (i < d_common.in_sqr_cols)
                sum = (sum + d_unique_d_in_sqr[(ei_new + (d_common.in_sqr_rows * i))])
                i += Int32(1)
            end
            d_unique_d_in_sqr_partial_sum[ei_new] = sum
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        if (tx == Int32(0))
            d_unique_d_in_final_sum[Int32(0)] = Int32(0)
            i = Int32(0)
            while (i < d_common.in_cols)
                d_unique_d_in_final_sum[Int32(0)] = (d_unique_d_in_final_sum[Int32(0)] + d_unique_d_in_partial_sum[i])
                i += Int32(1)
            end
        else
            if (tx == Int32(1))
                d_unique_d_in_sqr_final_sum[Int32(0)] = Int32(0)
                i = Int32(0)
                while (i < d_common.in_sqr_cols)
                    d_unique_d_in_sqr_final_sum[Int32(0)] = (d_unique_d_in_sqr_final_sum[Int32(0)] + d_unique_d_in_sqr_partial_sum[i])
                    i += Int32(1)
                end
            end
        end
        CUDA.sync_threads()
        if (tx == Int32(0))
            mean = (d_unique_d_in_final_sum[Int32(0)] / d_common.in_elem)
            mean_sqr = (mean * mean)
            variance = ((d_unique_d_in_sqr_final_sum[Int32(0)] / d_common.in_elem) - mean_sqr)
            deviation = sqrt(variance)
            d_unique_d_denomT[Int32(0)] = (sqrt(Float32((d_common.in_elem - Int32(1)))) * deviation)
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            d_unique_d_in2_sqr_sub2[ei_new] = (d_unique_d_in2_sqr_sub2[ei_new] * d_unique_d_denomT[Int32(0)])
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.conv_elem)
            d_unique_d_conv[ei_new] = (d_unique_d_conv[ei_new] - ((d_unique_d_in2_sub2[ei_new] * d_unique_d_in_final_sum[Int32(0)]) / d_common.in_elem))
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.in2_sub2_elem)
            d_unique_d_in2_sqr_sub2[ei_new] = (d_unique_d_conv[ei_new] / d_unique_d_in2_sqr_sub2[ei_new])
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        cent = ((d_common.sSize + d_common.tSize) + Int32(1))
        if (frame_no == Int32(0))
            tMask_row = (((cent + d_unique_d_Row[d_unique_point_no]) - d_unique_d_Row[d_unique_point_no]) - Int32(1))
            tMask_col = (((cent + d_unique_d_Col[d_unique_point_no]) - d_unique_d_Col[d_unique_point_no]) - Int32(1))
        else
            pointer = (((d_unique_point_no * d_common.no_frames) + frame_no) - Int32(1))
            tMask_row = (((cent + d_unique_d_tRowLoc[pointer]) - d_unique_d_Row[d_unique_point_no]) - Int32(1))
            tMask_col = (((cent + d_unique_d_tColLoc[pointer]) - d_unique_d_Col[d_unique_point_no]) - Int32(1))
        end
        ei_new = tx
        while (ei_new < d_common.tMask_elem)
            location = ((tMask_col * d_common.tMask_rows) + tMask_row)
            if (ei_new == location)
                d_unique_d_tMask[ei_new] = Int32(1)
            else
                d_unique_d_tMask[ei_new] = Int32(0)
            end
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.mask_conv_elem)
            ic = ((ei_new + Int32(1)) % d_common.mask_conv_rows)
            jc = (div((ei_new + Int32(1)),d_common.mask_conv_rows) + Int32(1))
            if (((ei_new + Int32(1)) % d_common.mask_conv_rows) == Int32(0))
                ic = d_common.mask_conv_rows
                jc = (jc - Int32(1))
            end
            j = (jc + d_common.mask_conv_joffset)
            jp1 = (j + Int32(1))
            if (d_common.mask_cols < jp1)
                ja1 = (jp1 - d_common.mask_cols)
            else
                ja1 = Int32(1)
            end
            if (d_common.tMask_cols < j)
                ja2 = d_common.tMask_cols
            else
                ja2 = j
            end
            i = (ic + d_common.mask_conv_ioffset)
            ip1 = (i + Int32(1))
            if (d_common.mask_rows < ip1)
                ia1 = (ip1 - d_common.mask_rows)
            else
                ia1 = Int32(1)
            end
            if (d_common.tMask_rows < i)
                ia2 = d_common.tMask_rows
            else
                ia2 = i
            end
            s = Int32(0)
            ja = ja1
            while (ja <= ja2)
                jb = (jp1 - ja)
                ia = ia1
                while (ia <= ia2)
                    ib = (ip1 - ia)
                    s = fma(Float32(d_unique_d_tMask[(((d_common.tMask_rows * (ja - Int32(1))) + ia) - Int32(1))]),Float32(Int32(1)),Float32(s))
                    ia += Int32(1)
                end
                ja += Int32(1)
            end
            d_unique_d_mask_conv[ei_new] = (d_unique_d_in2_sqr_sub2[ei_new] * s)
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        ei_new = tx
        while (ei_new < d_common.mask_conv_rows)
            i = Int32(0)
            while (i < d_common.mask_conv_cols)
                largest_coordinate_current = ((ei_new * d_common.mask_conv_rows) + i)
                largest_value_current = abs(d_unique_d_mask_conv[largest_coordinate_current])
                if (largest_value_current > largest_value)
                    largest_coordinate = largest_coordinate_current
                    largest_value = largest_value_current
                end
                i += Int32(1)
            end
            d_unique_d_par_max_coo[ei_new] = largest_coordinate
            d_unique_d_par_max_val[ei_new] = largest_value
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
        if (tx == Int32(0))
            i = Int32(0)
            while (i < d_common.mask_conv_rows)
                if (d_unique_d_par_max_val[i] > fin_max_val)
                    fin_max_val = d_unique_d_par_max_val[i]
                    fin_max_coo = d_unique_d_par_max_coo[i]
                end
                i += Int32(1)
            end
            largest_row = (((fin_max_coo + Int32(1)) % d_common.mask_conv_rows) - Int32(1))
            largest_col = div((fin_max_coo + Int32(1)),d_common.mask_conv_rows)
            if (((fin_max_coo + Int32(1)) % d_common.mask_conv_rows) == Int32(0))
                largest_row = (d_common.mask_conv_rows - Int32(1))
                largest_col = (largest_col - Int32(1))
            end
            largest_row = (largest_row + Int32(1))
            largest_col = (largest_col + Int32(1))
            offset_row = ((largest_row - d_common.in_rows) - (d_common.sSize - d_common.tSize))
            offset_col = ((largest_col - d_common.in_cols) - (d_common.sSize - d_common.tSize))
            pointer = ((d_unique_point_no * d_common.no_frames) + frame_no)
            d_unique_d_tRowLoc[pointer] = (d_unique_d_Row[d_unique_point_no] + offset_row)
            d_unique_d_tColLoc[pointer] = (d_unique_d_Col[d_unique_point_no] + offset_col)
        end
        CUDA.sync_threads()
    end
    if ((frame_no != Int32(0)) && ((frame_no % Int32(10)) == Int32(0)))
        loc_pointer = ((d_unique_point_no * d_common.no_frames) + frame_no)
        d_unique_d_Row[d_unique_point_no] = d_unique_d_tRowLoc[loc_pointer]
        d_unique_d_Col[d_unique_point_no] = d_unique_d_tColLoc[loc_pointer]
        ei_new = tx
        while (ei_new < d_common.in_elem)
            row = (((ei_new + Int32(1)) % d_common.in_rows) - Int32(1))
            col = ((div((ei_new + Int32(1)),d_common.in_rows) + Int32(1)) - Int32(1))
            if (((ei_new + Int32(1)) % d_common.in_rows) == Int32(0))
                row = (d_common.in_rows - Int32(1))
                col = (col - Int32(1))
            end
            ori_row = (((d_unique_d_Row[d_unique_point_no] - Int32(25)) + row) - Int32(1))
            ori_col = (((d_unique_d_Col[d_unique_point_no] - Int32(25)) + col) - Int32(1))
            ori_pointer = ((ori_col * d_common.frame_rows) + ori_row)
            d_in[ei_new] = fma(Float32((Int32(1) - d_common.alpha)),Float32(d_frame[ori_pointer]),Float32((d_common.alpha * d_in[ei_new])))
            ei_new = (ei_new + Int32(256))
        end
        CUDA.sync_threads()
    end
    nothing
end
