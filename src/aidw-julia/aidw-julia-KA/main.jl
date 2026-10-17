using CUDA, KernelAbstractions
const A1 = 1.5f0
const A2 = 2f0
const A3 = 2.5f0
const A4 = 3f0
const A5 = 3.5f0
const R_MIN = 0f0
const R_MAX = 2f0
const BLOCK_SIZE = 256
const EPS = 1f0

@kernel function aidw_ka!(dx,dy,dz,dnum,ix,iy,iz,inum,area,avg_dist)
    tid=@index(Global,Linear)
    if tid<=inum
    @inbounds r_obs = avg_dist[tid]
    r_exp = 0.5f0 / sqrt(Float32(dnum) / area)
    R_S0 = r_obs / r_exp

    u_R = 0f0
    if R_S0 >= R_MIN
        u_R = 0.5f0 - 0.5f0 * cos(3.1415926f0 / R_MAX * (R_S0 - R_MIN))
    end
    if R_S0 >= R_MAX
        u_R = 1f0
    end

    alpha = 1f0
    if u_R >= 0f0 && u_R <= 0.1f0;  alpha = A1;  end
    if u_R > 0.1f0 && u_R <= 0.3f0; alpha = A1 * (1f0 - 5f0*(u_R-0.1f0)) + A2 * 5f0*(u_R-0.1f0); end
    if u_R > 0.3f0 && u_R <= 0.5f0; alpha = A3 * 5f0*(u_R-0.3f0) + A1 * (1f0 - 5f0*(u_R-0.3f0)); end
    if u_R > 0.5f0 && u_R <= 0.7f0; alpha = A3 * (1f0 - 5f0*(u_R-0.5f0)) + A4 * 5f0*(u_R-0.5f0); end
    if u_R > 0.7f0 && u_R <= 0.9f0; alpha = A5 * 5f0*(u_R-0.7f0) + A4 * (1f0 - 5f0*(u_R-0.7f0)); end
    if u_R > 0.9f0 && u_R <= 1f0;   alpha = A5; end
    alpha *= 0.5f0

    sum = 0f0
    z = 0f0
    @inbounds ixt = ix[tid]
    @inbounds iyt = iy[tid]
    j = Int32(1)
    while j <= dnum
        @inbounds dxj = dx[j]
        @inbounds dyj = dy[j]
        @inbounds dzj = dz[j]
        ddx = ixt - dxj
        ddy = iyt - dyj
        dist = ddx*ddx + ddy*ddy
        t = 1f0 / (dist)^(alpha)
        sum += t
        z += dzj * t
        j += Int32(1)
    end
    @inbounds iz[tid] = z / sum
    end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
aidw_study((dx,dy,dz,dnum,ix,iy,iz,inum,area,avg)->()->aidw_ka!(CUDABackend(),256)(dx,dy,dz,dnum,ix,iy,iz,inum,area,avg;ndrange=Int(inum)))
