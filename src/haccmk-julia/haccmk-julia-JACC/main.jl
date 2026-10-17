using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function haccmk_jacc!(index,n1::Int32, n2::Int32, xx, yy, zz, mass, vx2, vy2, vz2,
                        fsrmax::Float32, mp_rsm::Float32, fcoeff::Float32)
    i0 = Int32(index)-Int32(1)
    if i0 < n1

    ma0 = Float32(0.269327)
    ma1 = Float32(-0.0750978)
    ma2 = Float32(0.0114808)
    ma3 = Float32(-0.00109313)
    ma4 = Float32(0.0000605491)
    ma5 = Float32(-0.00000147177)

    xi = Float32(0)
    yi = Float32(0)
    zi = Float32(0)
    @inbounds xxi = xx[Int(i0) + 1]
    @inbounds yyi = yy[Int(i0) + 1]
    @inbounds zzi = zz[Int(i0) + 1]

    for j0 in Int32(0):(n2 - Int32(1))
        @inbounds dxc = xx[Int(j0) + 1] - xxi
        @inbounds dyc = yy[Int(j0) + 1] - yyi
        @inbounds dzc = zz[Int(j0) + 1] - zzi
        r2 = dxc * dxc + dyc * dyc + dzc * dzc
        @inbounds m = mass[Int(j0) + 1] * Float32(r2 < fsrmax)
        f = r2 + mp_rsm
        f = m * (Float32(1) / (f * sqrt(f)) -
                 (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))))
        xi += f * dxc
        yi += f * dyc
        zi += f * dzc
    end

    @inbounds vx2[Int(i0) + 1] += xi * fcoeff
    @inbounds vy2[Int(i0) + 1] += yi * fcoeff
    @inbounds vz2[Int(i0) + 1] += zi * fcoeff
    end
end

function factory(x,y,z,mass,vx,vy,vz,n1,n2,cut,soft,fc)
 spec=JACC.launch_spec(;threads=256,blocks=4,shmem_size=0,sync=false)
 return ()->JACC.parallel_for(spec,n1,haccmk_jacc!,Int32(n1),Int32(n2),x,y,z,mass,vx,vy,vz,cut,soft,fc)
end
hacc_study(factory)
