const R00 = Float32(-0.9111348390579224)
const R01 = Float32( 0.0751304179430008)
const R02 = Float32(-0.4052018225193024)
const R10 = Float32(-0.3360927104949951)
const R11 = Float32(-0.7044632434844971)
const R12 = Float32( 0.6251187324523926)
const R20 = Float32(-0.2384843230247498)
const R21 = Float32( 0.7057529091835022)
const R22 = Float32( 0.6671117544174194)
const T0 = 0.1f0; const T1 = -0.2f0; const T2 = 0.3f0
const D0 = 0.03f0; const D1 = -0.15f0; const D2 = 0.001f0; const D3 = 0.001f0; const D4 = 0.05f0
const FX = 1200.0f0; const CX = 400.0f0; const FY = 1200.0f0; const CY = 300.0f0
const POINT_STEP = 8

function project_item!(i,cp,ox,oy,oz,ov,n)
    if i > n
        return
    end
    base = (i - Int32(1)) * Int32(POINT_STEP) + Int32(1)
    @inbounds p0 = cp[base]
    @inbounds p1 = cp[base + 1]
    @inbounds p2 = cp[base + 2]
    pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02
    pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12
    pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22
    close = pt2 > 2.5f0
    denom = close ? pt2 : 1.0f0
    tmpx = pt0 / denom
    tmpy = pt1 / denom
    r2 = tmpx * tmpx + tmpy * tmpy
    tmpdist = 1.0f0 + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2
    ix = tmpx * tmpdist + 2.0f0 * D2 * tmpx * tmpy + D3 * (r2 + 2.0f0 * tmpx * tmpx)
    iy = tmpy * tmpdist + D2 * (r2 + 2.0f0 * tmpy * tmpy) + 2.0f0 * D3 * tmpx * tmpy
    ux = FX * ix + CX
    uy = FY * iy + CY
    xpix = ux + 0.5f0
    ypix = uy + 0.5f0
    @inbounds if close
        ox[i] = xpix; oy[i] = ypix; oz[i] = pt2 * 100.0f0; ov[i] = Int32(1)
    else
        ox[i] = 0.0f0; oy[i] = 0.0f0; oz[i] = 0.0f0; ov[i] = Int32(0)
    end
    return
end
function ref_project(p0::Float32, p1::Float32, p2::Float32)
    pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02
    pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12
    pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22
    close = pt2 > 2.5f0
    denom = close ? pt2 : 1.0f0
    tmpx = pt0 / denom
    tmpy = pt1 / denom
    r2 = tmpx * tmpx + tmpy * tmpy
    tmpdist = 1.0f0 + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2
    ix = tmpx * tmpdist + 2.0f0 * D2 * tmpx * tmpy + D3 * (r2 + 2.0f0 * tmpx * tmpx)
    iy = tmpy * tmpdist + D2 * (r2 + 2.0f0 * tmpy * tmpy) + 2.0f0 * D3 * tmpx * tmpy
    xpix = FX * ix + CX + 0.5f0
    ypix = FY * iy + CY + 0.5f0
    if close
        return xpix, ypix, pt2 * 100.0f0, Int32(1)
    else
        return 0.0f0, 0.0f0, 0.0f0, Int32(0)
    end
end

