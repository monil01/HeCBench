include(joinpath(@__DIR__,"study_constants.jl"))

@inline function update_state_z_body!(state, nx::Int32, nz::Int32, hs::Int32, i::Int32, ll::Int32)
    if i < nx + 2hs && ll < NUM_VARS
        if ll == ID_WMOM
            state[state_idx(ll, Int32(0), i, nx, nz, hs)] = 0.0
            state[state_idx(ll, Int32(1), i, nx, nz, hs)] = 0.0
            state[state_idx(ll, nz + hs, i, nx, nz, hs)] = 0.0
            state[state_idx(ll, nz + hs + Int32(1), i, nx, nz, hs)] = 0.0
        else
            state[state_idx(ll, Int32(0), i, nx, nz, hs)] = state[state_idx(ll, hs, i, nx, nz, hs)]
            state[state_idx(ll, Int32(1), i, nx, nz, hs)] = state[state_idx(ll, hs, i, nx, nz, hs)]
            state[state_idx(ll, nz + hs, i, nx, nz, hs)] = state[state_idx(ll, nz + hs - Int32(1), i, nx, nz, hs)]
            state[state_idx(ll, nz + hs + Int32(1), i, nx, nz, hs)] = state[state_idx(ll, nz + hs - Int32(1), i, nx, nz, hs)]
        end
    end
    return
end

@inline function compute_flux_x_body!(state, flux, hy_dens_cell, hy_dens_theta_cell, hv_coef::Float64, nx::Int32, nz::Int32, hs::Int32, i::Int32, k::Int32)
    if i < nx + Int32(1) && k < nz
        stencils=ntuple(Val(4)) do index
            ll=Int32(index-1)
            s0 = state[state_idx(ll, k + hs, i, nx, nz, hs)]
            s1 = state[state_idx(ll, k + hs, i + Int32(1), nx, nz, hs)]
            s2 = state[state_idx(ll, k + hs, i + Int32(2), nx, nz, hs)]
            s3 = state[state_idx(ll, k + hs, i + Int32(3), nx, nz, hs)]
            (-s0 / 12.0 + 7.0 * s1 / 12.0 + 7.0 * s2 / 12.0 - s3 / 12.0, -s0 + 3.0 * s1 - 3.0 * s2 + s3)
        end
        vals=ntuple(j->stencils[j][1],Val(4));d3=ntuple(j->stencils[j][2],Val(4))
        r = vals[Int(ID_DENS) + 1] + hy_dens_cell[Int(k + hs) + 1]
        u = vals[Int(ID_UMOM) + 1] / r
        w = vals[Int(ID_WMOM) + 1] / r
        t = (vals[Int(ID_RHOT) + 1] + hy_dens_theta_cell[Int(k + hs) + 1]) / r
        p = C0 * (r * t)^GAMM
        flux[flux_idx(ID_DENS, k, i, nx, nz)] = r * u - hv_coef * d3[Int(ID_DENS) + 1]
        flux[flux_idx(ID_UMOM, k, i, nx, nz)] = r * u * u + p - hv_coef * d3[Int(ID_UMOM) + 1]
        flux[flux_idx(ID_WMOM, k, i, nx, nz)] = r * u * w - hv_coef * d3[Int(ID_WMOM) + 1]
        flux[flux_idx(ID_RHOT, k, i, nx, nz)] = r * u * t - hv_coef * d3[Int(ID_RHOT) + 1]
    end
    return
end

@inline function compute_tend_x_body!(flux, tend, nx::Int32, nz::Int32, dx::Int32, i::Int32, k::Int32, ll::Int32)
    if i < nx && k < nz && ll < NUM_VARS
        tend[tend_idx(ll, k, i, nx, nz)] = -(flux[flux_idx(ll, k, i + Int32(1), nx, nz)] - flux[flux_idx(ll, k, i, nx, nz)]) / dx
    end
    return
end

@inline function compute_flux_z_body!(state, flux, hy_dens_int, hy_pressure_int, hy_dens_theta_int, hv_coef::Float64, nx::Int32, nz::Int32, hs::Int32, i::Int32, k::Int32)
    if i < nx && k < nz + Int32(1)
        stencils=ntuple(Val(4)) do index
            ll=Int32(index-1)
            s0 = state[state_idx(ll, k, i + hs, nx, nz, hs)]
            s1 = state[state_idx(ll, k + Int32(1), i + hs, nx, nz, hs)]
            s2 = state[state_idx(ll, k + Int32(2), i + hs, nx, nz, hs)]
            s3 = state[state_idx(ll, k + Int32(3), i + hs, nx, nz, hs)]
            (-s0 / 12.0 + 7.0 * s1 / 12.0 + 7.0 * s2 / 12.0 - s3 / 12.0, -s0 + 3.0 * s1 - 3.0 * s2 + s3)
        end
        vals=ntuple(j->stencils[j][1],Val(4));d3=ntuple(j->stencils[j][2],Val(4))
        r = vals[Int(ID_DENS) + 1] + hy_dens_int[Int(k) + 1]
        u = vals[Int(ID_UMOM) + 1] / r
        w = vals[Int(ID_WMOM) + 1] / r
        t = (vals[Int(ID_RHOT) + 1] + hy_dens_theta_int[Int(k) + 1]) / r
        p = C0 * (r * t)^GAMM - hy_pressure_int[Int(k) + 1]
        if k == Int32(0) || k == nz
            w = 0.0
            d3=Base.setindex(d3,0.0,1)
        end
        flux[flux_idx(ID_DENS, k, i, nx, nz)] = r * w - hv_coef * d3[Int(ID_DENS) + 1]
        flux[flux_idx(ID_UMOM, k, i, nx, nz)] = r * w * u - hv_coef * d3[Int(ID_UMOM) + 1]
        flux[flux_idx(ID_WMOM, k, i, nx, nz)] = r * w * w + p - hv_coef * d3[Int(ID_WMOM) + 1]
        flux[flux_idx(ID_RHOT, k, i, nx, nz)] = r * w * t - hv_coef * d3[Int(ID_RHOT) + 1]
    end
    return
end

@inline function compute_tend_z_body!(state, flux, tend, nx::Int32, nz::Int32, hs::Int32, dz::Int32, i::Int32, k::Int32, ll::Int32)
    if i < nx && k < nz && ll < NUM_VARS
        v = -(flux[flux_idx(ll, k + Int32(1), i, nx, nz)] - flux[flux_idx(ll, k, i, nx, nz)]) / dz
        if ll == ID_WMOM
            v -= state[state_idx(ID_DENS, k + hs, i + hs, nx, nz, hs)] * GRAV
        end
        tend[tend_idx(ll, k, i, nx, nz)] = v
    end
    return
end

@inline function update_fluid_state_body!(state_init, state_out, tend, nx::Int32, nz::Int32, hs::Int32, dt::Float64, i::Int32, k::Int32, ll::Int32)
    if i < nx && k < nz && ll < NUM_VARS
        state_out[state_idx(ll, k + hs, i + hs, nx, nz, hs)] = state_init[state_idx(ll, k + hs, i + hs, nx, nz, hs)] + dt * tend[tend_idx(ll, k, i, nx, nz)]
    end
    return
end
@inline function pack_body!(state,sl,sr,nx::Int32,nz::Int32,hs::Int32,s::Int32,k::Int32,ll::Int32)
 if s<hs && k<nz && ll<4
  @inbounds sl[ll*nz*hs+k*hs+s+1]=state[state_idx(ll,k+hs,hs+s,nx,nz,hs)]
  @inbounds sr[ll*nz*hs+k*hs+s+1]=state[state_idx(ll,k+hs,nx+s,nx,nz,hs)]
 end
 return
end
@inline function unpack_body!(state,rl,rr,nx::Int32,nz::Int32,hs::Int32,s::Int32,k::Int32,ll::Int32)
 if s<hs && k<nz && ll<4
  @inbounds state[state_idx(ll,k+hs,s,nx,nz,hs)]=rl[ll*nz*hs+k*hs+s+1]
  @inbounds state[state_idx(ll,k+hs,nx+hs+s,nx,nz,hs)]=rr[ll*nz*hs+k*hs+s+1]
 end
 return
end
@inline function mass_terms_body(state,hy_dens_cell,hy_dens_theta_cell,nx::Int32,nz::Int32,dx::Float64,dz::Float64,i::Int32,k::Int32)
 r=state[state_idx(ID_DENS,k+HS,i+HS,nx,nz,HS)]+hy_dens_cell[Int(k+HS)+1]
 u=state[state_idx(ID_UMOM,k+HS,i+HS,nx,nz,HS)]/r
 w=state[state_idx(ID_WMOM,k+HS,i+HS,nx,nz,HS)]/r
 th=(state[state_idx(ID_RHOT,k+HS,i+HS,nx,nz,HS)]+hy_dens_theta_cell[Int(k+HS)+1])/r
 p=C0*(r*th)^GAMM;temp=th/(P0/p)^(RD/CP)
 return r*dx*dz,(r*(u*u+w*w)+r*CV*temp)*dx*dz
end
