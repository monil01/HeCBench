# Generated from configured/kernels.cu; source AST/source SHA in device_generation.json.
@inline mm_atomic!(v,i,x) = (CUDA.atomic_add!(pointer(v,Int(i)+1),x);nothing)
@inline function mm_body0(vals,fbrLikeSlcInds,dInds2,fbrPtr0,fbrPtr1,fbrIdx1,nFibers,dU0,dU1,dU2,mode,R,warpPerSlice,logOfWPC,fbrPerWarp,logOfFPW,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbr = Int64(((gId >> (Int64(5) + logOfWPC)) << logOfFPW))
    tmp = Float32(Int64(0))
    tmp_val = Float32(0)
    if (fbr < (nFibers - Int64(1)))
        tmp_val = Float32(Int64(0))
        diffFiber = Bool(false)
        idx0 = Int64(0)
        fr = Int64(Int64(0))
        while ((fr < fbrPerWarp) && ((fbr + fr) < (nFibers - Int64(1))))
            diffFiber = Bool(false)
            idx1 = Int64(fbrIdx1[Int((fbr + fr))+1])
            idx0 = Int64(fbrLikeSlcInds[Int((fbr + fr))+1])
            tmp_val = Float32(Int64(0))
            x = Int64((fbrPtr1[Int((fbr + fr))+1] + workId))
            while (x < fbrPtr1[Int(((fbr + fr) + Int64(1)))+1])
                idx2 = Int64(dInds2[Int(x)+1])
                r = Int64(laneId)
                while (r < R)
                    tmp_val = muladd(vals[Int(x)+1],dU2[Int(((idx2 * R) + r))+1],tmp_val)
                    r += Int64(32)
                end
                x += warpPerSlice
            end
            r = Int64(laneId)
            while (r < R)
                tmp = muladd(tmp_val,dU1[Int(((idx1 * R) + r))+1],tmp)
                r += Int64(32)
            end
            if (fbrLikeSlcInds[Int((fbr + fr))+1] != fbrLikeSlcInds[Int(((fbr + fr) + Int64(1)))+1])
                diffFiber = Bool(true)
                r = Int64(laneId)
                while (r < R)
                    mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp))
                    r += Int64(32)
                end
                tmp = Float32(Int64(0))
            end
            fr += 1
        end
        if (!diffFiber)
            r = Int64(laneId)
            while (r < R)
                mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp))
                r += Int64(32)
            end
        end
    end
    end
    nothing
end
@inline function mm_body1(vals,fbrLikeSlcInds,dInds3,fbrPtr0,fbrPtr1,fbrIdx1,fbrPtr2,fbrIdx2,nFibers,dU0,dU1,dU2,dU3,mode,R,warpPerSlice,logOfWPC,fbrPerWarp,logOfFPW,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbrS = Int64(((gId >> (Int64(5) + logOfWPC)) << logOfFPW))
    tmp = Float32(Int64(0))
    tmp_val = Float32(0)
    tmp2 = Float32(Int64(0))
    if (fbrS < (nFibers - Int64(1)))
        tmp_val = Float32(Int64(0))
        diffFiber = Bool(false)
        idx0 = Int64(0)
        fr = Int64(Int64(0))
        while ((fr < fbrPerWarp) && ((fbrS + fr) < (nFibers - Int64(1))))
            diffFiber = Bool(false)
            idx1 = Int64(fbrIdx1[Int((fbrS + fr))+1])
            idx0 = Int64(fbrLikeSlcInds[Int((fbrS + fr))+1])
            tmp = Float32(Int64(0))
            fbr = Int64((fbrPtr1[Int((fbrS + fr))+1] + workId))
            while (fbr < fbrPtr1[Int(((fbrS + fr) + Int64(1)))+1])
                idx2 = Int64(fbrIdx2[Int(fbr)+1])
                tmp_val = Float32(Int64(0))
                x = Int64(fbrPtr2[Int(fbr)+1])
                while (x < fbrPtr2[Int((fbr + Int64(1)))+1])
                    idx3 = Int64(dInds3[Int(x)+1])
                    r = Int64(laneId)
                    while (r < R)
                        tmp_val = muladd(vals[Int(x)+1],dU3[Int(((idx3 * R) + r))+1],tmp_val)
                        r += Int64(32)
                    end
                    x += 1
                end
                r = Int64(laneId)
                while (r < R)
                    tmp = muladd(tmp_val,dU2[Int(((idx2 * R) + r))+1],tmp)
                    r += Int64(32)
                end
                fbr += warpPerSlice
            end
            r = Int64(laneId)
            while (r < R)
                tmp2 = muladd(tmp,dU1[Int(((idx1 * R) + r))+1],tmp2)
                r += Int64(32)
            end
            if (fbrLikeSlcInds[Int((fbrS + fr))+1] != fbrLikeSlcInds[Int(((fbrS + fr) + Int64(1)))+1])
                diffFiber = Bool(true)
                r = Int64(laneId)
                while (r < R)
                    mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp2))
                    r += Int64(32)
                end
                tmp2 = Float32(Int64(0))
            end
            fr += 1
        end
        if (!diffFiber)
            r = Int64(laneId)
            while (r < R)
                mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp2))
                r += Int64(32)
            end
        end
    end
    end
    nothing
end
@inline function mm_body2(vals,fbrLikeSlcInds,dInds3,fbrPtr0,fbrPtr1,fbrIdx1,fbrPtr2,fbrIdx2,nFibers,dU0,dU1,dU2,dU3,mode,R,warpPerSlice,logOfWPC,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbrS = Int64((gId >> (Int64(5) + logOfWPC)))
    tmp = Float32(Int64(0))
    tmp_val = Float32(0)
    tmp2 = Float32(Int64(0))
    if (fbrS < (nFibers - Int64(1)))
        tmp = Float32(Int64(0))
        idx0 = Int64(fbrIdx1[Int(fbrS)+1])
        idx3 = Int64(fbrLikeSlcInds[Int(fbrS)+1])
        fbr = Int64((fbrPtr1[Int(fbrS)+1] + workId))
        while (fbr < fbrPtr1[Int((fbrS + Int64(1)))+1])
            idx1 = Int64(fbrIdx2[Int(fbr)+1])
            tmp_val = Float32(Int64(0))
            x = Int64(fbrPtr2[Int(fbr)+1])
            while (x < fbrPtr2[Int((fbr + Int64(1)))+1])
                idx2 = Int64(dInds3[Int(x)+1])
                r = Int64(laneId)
                while (r < R)
                    tmp_val = muladd(vals[Int(x)+1],dU2[Int(((idx2 * R) + r))+1],tmp_val)
                    r += Int64(32)
                end
                x += 1
            end
            r = Int64(laneId)
            while (r < R)
                tmp = muladd(tmp_val,dU1[Int(((idx1 * R) + r))+1],tmp)
                r += Int64(32)
            end
            fbr += warpPerSlice
        end
        r = Int64(laneId)
        while (r < R)
            tmp2 = Float32((tmp * dU3[Int(((idx3 * R) + r))+1]))
            mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp2))
            r += Int64(32)
        end
    end
    end
    nothing
end
@inline function mm_body3(vals,fbrLikeSlcInds,dInds2,fbrPtr0,fbrPtr1,fbrIdx1,nFibers,dU0,dU1,dU2,mode,R,warpPerSlice,logOfWPC,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbr = Int64((gId >> (Int64(5) + logOfWPC)))
    tmp = Float32(Int64(0))
    tmp_val = Float32(0)
    if (fbr < (nFibers - Int64(1)))
        tmp_val = Float32(Int64(0))
        idx0 = Int64(fbrIdx1[Int(fbr)+1])
        idx2 = Int64(fbrLikeSlcInds[Int(fbr)+1])
        x = Int64((fbrPtr1[Int(fbr)+1] + workId))
        while (x < fbrPtr1[Int((fbr + Int64(1)))+1])
            idx1 = Int64(dInds2[Int(x)+1])
            r = Int64(laneId)
            while (r < R)
                tmp_val = muladd(vals[Int(x)+1],dU1[Int(((idx1 * R) + r))+1],tmp_val)
                r += Int64(32)
            end
            x += warpPerSlice
        end
        r = Int64(laneId)
        while (r < R)
            tmp = Float32((tmp_val * dU2[Int(((idx2 * R) + r))+1]))
            mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp))
            r += Int64(32)
        end
    end
    end
    nothing
end
@inline function mm_body4(vals,fbrLikeSlcInds,dInds3,fbrPtr0,fbrPtr1,fbrIdx1,fbrPtr2,fbrIdx2,nFibers,dU0,dU1,dU2,dU3,mode,R,warpPerSlice,logOfWPC,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbrS = Int64((gId >> (Int64(5) + logOfWPC)))
    tmp = Float32(0)
    if (fbrS < (nFibers - Int64(1)))
        idx2 = Int64(fbrLikeSlcInds[Int(fbrS)+1])
        idx3 = Int64(fbrIdx1[Int(fbrS)+1])
        fbr = Int64((fbrPtr1[Int(fbrS)+1] + workId))
        while (fbr < fbrPtr1[Int((fbrS + Int64(1)))+1])
            idx0 = Int64(fbrIdx2[Int(fbr)+1])
            tmp = Float32(Int64(0))
            x = Int64(fbrPtr2[Int(fbr)+1])
            while (x < fbrPtr2[Int((fbr + Int64(1)))+1])
                idx1 = Int64(dInds3[Int(x)+1])
                r = Int64(laneId)
                while (r < R)
                    tmp = muladd(vals[Int(x)+1],dU1[Int(((idx1 * R) + r))+1],tmp)
                    r += Int64(32)
                end
                x += 1
            end
            r = Int64(laneId)
            while (r < R)
                mm_atomic!(dU0,((idx0 * R) + r),Float64(((tmp * dU2[Int(((idx2 * R) + r))+1]) * dU3[Int(((idx3 * R) + r))+1])))
                r += Int64(32)
            end
            fbr += warpPerSlice
        end
    end
    end
    nothing
end
@inline function mm_body5(vals,fbrLikeSlcInds,dInds2,fbrPtr0,fbrPtr1,fbrIdx1,nFibers,dU0,dU1,dU2,mode,R,warpPerSlice,logOfWPC,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbr = Int64((gId >> (Int64(5) + logOfWPC)))
    tmp = Float32(Int64(0))
    tmp_val = Float32(0)
    if (fbr < (nFibers - Int64(1)))
        tmp_val = Float32(Int64(0))
        idx1 = Int64(fbrLikeSlcInds[Int(fbr)+1])
        idx2 = Int64(fbrIdx1[Int(fbr)+1])
        r = Int64(laneId)
        while (r < R)
            tmp = Float32((dU1[Int(((idx1 * R) + r))+1] * dU2[Int(((idx2 * R) + r))+1]))
            r += Int64(32)
        end
        x = Int64((fbrPtr1[Int(fbr)+1] + workId))
        while (x < fbrPtr1[Int((fbr + Int64(1)))+1])
            idx0 = Int64(dInds2[Int(x)+1])
            r = Int64(laneId)
            while (r < R)
                tmp_val = Float32((vals[Int(x)+1] * tmp))
                mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp_val))
                r += Int64(32)
            end
            x += warpPerSlice
        end
    end
    end
    nothing
end
@inline function mm_body6(vals,fbrLikeSlcInds,dInds3,fbrPtr0,fbrPtr1,fbrIdx1,fbrPtr2,fbrIdx2,nFibers,dU0,dU1,dU2,dU3,mode,R,warpPerSlice,logOfWPC,tx,bx,bd)
    @inbounds begin
    tId = Int64(tx)
    laneId = Int64((tId & Int64(31)))
    bdim = Int64(bd)
    gId = Int64(((bx * bdim) + tId))
    workId = Int64(((tId & ((Int64(1) << (Int64(5) + logOfWPC)) - Int64(1))) >> Int64(5)))
    fbrS = Int64((gId >> (Int64(5) + logOfWPC)))
    tmp = Float32(Int64(0))
    tmp_val = Float32(Int64(0))

    if (fbrS < (nFibers - Int64(1)))
        tmp = Float32(Int64(0))
        idx1 = Int64(fbrLikeSlcInds[Int(fbrS)+1])
        idx2 = Int64(fbrIdx1[Int(fbrS)+1])
        r = Int64(laneId)
        while (r < R)
            tmp_val = Float32((dU1[Int(((idx1 * R) + r))+1] * dU2[Int(((idx2 * R) + r))+1]))
            r += Int64(32)
        end
        fbr = Int64((fbrPtr1[Int(fbrS)+1] + workId))
        while (fbr < fbrPtr1[Int((fbrS + Int64(1)))+1])
            idx3 = Int64(fbrIdx2[Int(fbr)+1])
            x = Int64(fbrPtr2[Int(fbr)+1])
            while (x < fbrPtr2[Int((fbr + Int64(1)))+1])
                idx0 = Int64(dInds3[Int(x)+1])
                r = Int64(laneId)
                while (r < R)
                    tmp = Float32(((vals[Int(x)+1] * dU3[Int(((idx3 * R) + r))+1]) * tmp_val))
                    mm_atomic!(dU0,((idx0 * R) + r),Float64(tmp))
                    r += Int64(32)
                end
                x += 1
            end
            fbr += warpPerSlice
        end
    end
    end
    nothing
end
