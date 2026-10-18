# Derived from SW4CK GPL-2.0-only source; see LICENSE and source_translation_manifest.json.
function cpu_kernel1!(start0,N0,start1,N1,start2,N2,ifirst,ilast,jfirst,jlast,kfirst,klast,a1,sgn,a_u,a_mu,a_lambda,a_met,a_jac,a_lu,a_acof,a_bope,a_ghcof,a_acof_no_gp,a_ghcof_no_gp,a_strx,a_stry)
ni=ilast-ifirst+1
nij=ni*(jlast-jfirst+1)
nijk=nij*(klast-kfirst+1)
base=-(ifirst+ni*jfirst+nij*kfirst)
i6=1.0/6.0;tf=0.75;c1=2.0/3.0;c2=-1.0/12.0
i = Int32(start0)
j = Int32(start1)
k = Int32(start2)
if (i < N0) && (j < N1) && (k < N2)
ijac_s1 = Float64(a_strx[(i)-ifirst+1] * a_stry[(j)-jfirst+1] / a_jac[base+(i)+ni*( j)+nij*( k)+1])
istry_s2 = Float64(1 / (a_stry[(j)-jfirst+1]))
istrx_s3 = Float64(1 / (a_strx[(i)-ifirst+1]))
istrxy_s4 = Float64(istry_s2 * istrx_s3)
r1_s5 = Float64(0)
r2_s6 = Float64(0)
r3_s7 = Float64(0)
cof1_s8 = Float64((2 * a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 2)-ifirst+1])
cof2_s9 = Float64((2 * a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 1)-ifirst+1])
cof3_s10 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s11 = Float64((2 * a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 1)-ifirst+1])
cof5_s12 = Float64((2 * a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 2)-ifirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r1_s5 = Float64(r1_s5 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux2_s14 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux3_s15 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux4_s16 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
      istry_s2)
cof1_s8 = Float64((a_mu[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s9 = Float64((a_mu[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s11 = Float64((a_mu[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s12 = Float64((a_mu[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r1_s5 = Float64(r1_s5 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux2_s14 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux3_s15 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux4_s16 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
      istrx_s3)
cof1_s8 = Float64((a_mu[base+(i - 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 2)-ifirst+1])
cof2_s9 = Float64((a_mu[base+(i - 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 1)-ifirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s11 = Float64((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 1)-ifirst+1])
cof5_s12 = Float64((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 2)-ifirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r2_s6 = Float64(r2_s6 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s14 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s15 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s16 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istry_s2)
cof1_s8 = Float64((2 * a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1]) *
      a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s9 = Float64((2 * a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1]) *
      a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s10 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s11 = Float64((2 * a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1]) *
      a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s12 = Float64((2 * a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1]) *
      a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r2_s6 = Float64(r2_s6 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s14 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s15 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s16 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istrx_s3)
cof1_s8 = Float64((a_mu[base+(i - 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 2)-ifirst+1])
cof2_s9 = Float64((a_mu[base+(i - 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 1)-ifirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s11 = Float64((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 1)-ifirst+1])
cof5_s12 = Float64((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 2)-ifirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r3_s7 = Float64(r3_s7 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s14 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s15 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s16 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istry_s2)
cof1_s8 = Float64((a_mu[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s9 = Float64((a_mu[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s11 = Float64((a_mu[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s12 = Float64((a_mu[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r3_s7 = Float64(r3_s7 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s14 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s15 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s16 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istrx_s3)
local mucofu2_s17
local mucofuv_s18
local mucofuw_s19
local mucofvw_s20
local mucofv2_s21
local mucofw2_s22
for q_s23 in Int32(1):Int32(8)
mucofu2_s17 = Float64(0)
mucofuv_s18 = Float64(0)
mucofuw_s19 = Float64(0)
mucofvw_s20 = Float64(0)
mucofv2_s21 = Float64(0)
mucofw2_s22 = Float64(0)
for m_s24 in Int32(1):Int32(8)
mucofu2_s17 += (a_acof[(k)-1+6*(( q_s23)-1)+48*(( m_s24)-1)+1] *
          ((2 * a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] *
           a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
           a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])))
mucofv2_s21 += (a_acof[(k)-1+6*(( q_s23)-1)+48*(( m_s24)-1)+1] *
          ((2 * a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] *
           a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
           a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])))
mucofw2_s22 += (a_acof[(k)-1+6*(( q_s23)-1)+48*(( m_s24)-1)+1] *
          ((2 * a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] +
           a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1])))
mucofuv_s18 += (a_acof[(k)-1+6*(( q_s23)-1)+48*(( m_s24)-1)+1] * (a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) *
          a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1])
mucofuw_s19 += (a_acof[(k)-1+6*(( q_s23)-1)+48*(( m_s24)-1)+1] * (a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) *
          a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])
mucofvw_s20 += (a_acof[(k)-1+6*(( q_s23)-1)+48*(( m_s24)-1)+1] * (a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) *
          a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])
end
r1_s5 += (istrxy_s4 * mucofu2_s17 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(1)+1] + mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(2)+1] +
        istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(3)+1])
r2_s6 += (mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(1)+1] + istrxy_s4 * mucofv2_s21 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(2)+1] +
        istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(3)+1])
r3_s7 += (istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(1)+1] +
        istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(2)+1] +
        istrxy_s4 * mucofw2_s22 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(3)+1])
end
mucofu2_s17 = Float64(a_ghcof[(k)] * ((2 * a_mu[base+(i)+ni*( j)+nij*( 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] *
          a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
          a_mu[base+(i)+ni*( j)+nij*( 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1])))
mucofv2_s21 = Float64(a_ghcof[(k)] * ((2 * a_mu[base+(i)+ni*( j)+nij*( 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] *
          a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
          a_mu[base+(i)+ni*( j)+nij*( 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1])))
mucofw2_s22 = Float64(a_ghcof[(k)] *
      ((2 * a_mu[base+(i)+ni*( j)+nij*( 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1] *
       a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1] +
       a_mu[base+(i)+ni*( j)+nij*( 1)+1] *
       (a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1])))
mucofuv_s18 = Float64(a_ghcof[(k)] * (a_mu[base+(i)+ni*( j)+nij*( 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1])
mucofuw_s19 = Float64(a_ghcof[(k)] * (a_mu[base+(i)+ni*( j)+nij*( 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1])
mucofvw_s20 = Float64(a_ghcof[(k)] * (a_mu[base+(i)+ni*( j)+nij*( 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(3)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( 1)+nijk*(4)+1])
r1_s5 += (istrxy_s4 * mucofu2_s17 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(1)+1] + mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(2)+1] +
      istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(3)+1])
r2_s6 += (mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(1)+1] + istrxy_s4 * mucofv2_s21 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(2)+1] +
      istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(3)+1])
r3_s7 += (istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(1)+1] +
      istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(2)+1] +
      istrxy_s4 * mucofw2_s22 * a_u[base-nijk+( i)+ni*( j)+nij*( 0)+nijk*(3)+1])
r1_s5 += (c2 *
      (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1])) -
       a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1] -
          a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]))) +
      c1 *
      (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1])) -
       a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))))
r1_s5 += (c2 *
      (a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 *
        (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1])) -
       a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] -
          a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))) +
      c1 *
      (a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1])) -
       a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 *
        (a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))))
r2_s6 += (c2 *
      (a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1])) -
       a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1] -
          a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]))) +
      c1 *
      (a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1])) -
       a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))))
r2_s6 += (c2 *
      (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 *
        (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1])) -
       a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] -
          a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))) +
      c1 *
      (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 *
        (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1])) -
       a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 *
        (a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))))
dudrm2_s25 = Float64(0)
dudrm1_s26 = Float64(0)
dudrp1_s27 = Float64(0)
dudrp2_s28 = Float64(0)
dvdrm2_s29 = Float64(0)
dvdrm1_s30 = Float64(0)
dvdrp1_s31 = Float64(0)
dvdrp2_s32 = Float64(0)
dwdrm2_s33 = Float64(0)
dwdrm1_s34 = Float64(0)
dwdrp1_s35 = Float64(0)
dwdrp2_s36 = Float64(0)
for q_s37 in Int32(1):Int32(8)
dudrm2_s25 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrm2_s29 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrm2_s33 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
dudrm1_s26 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrm1_s30 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrm1_s34 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
dudrp2_s28 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrp2_s32 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrp2_s36 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
dudrp1_s27 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrp1_s31 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrp1_s35 += (a_bope[(k)-1+6*(( q_s37)-1)+1] * a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
end
r1_s5 += ((c2 * ((2 * a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1]) *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
          a_strx[(i + 2)-ifirst+1] * dudrp2_s28 +
          a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 * a_stry[(j)-jfirst+1] +
          a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 -
          ((2 * a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1]) *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
           a_strx[(i - 2)-ifirst+1] * dudrm2_s25 +
           a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm2_s29 * a_stry[(j)-jfirst+1] +
           a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm2_s33)) +
        c1 * ((2 * a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1]) *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
          a_strx[(i + 1)-ifirst+1] * dudrp1_s27 +
          a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 * a_stry[(j)-jfirst+1] +
          a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 -
          ((2 * a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1]) *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
           a_strx[(i - 1)-ifirst+1] * dudrm1_s26 +
           a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm1_s30 * a_stry[(j)-jfirst+1] +
           a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm1_s34))) *
           istry_s2)
r2_s6 += (c2 *
      (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
       a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp2_s28 +
       a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
       a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 * a_strx[(i + 2)-ifirst+1] * istry_s2 -
       (a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm2_s25 +
        a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm2_s29 * a_strx[(i - 2)-ifirst+1] * istry_s2)) +
      c1 * (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp1_s27 +
          a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 * a_strx[(i + 1)-ifirst+1] * istry_s2 -
          (a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm1_s26 +
           a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm1_s30 * a_strx[(i - 1)-ifirst+1] * istry_s2)))
r3_s7 += (istry_s2 *
      (c2 * (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
             a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp2_s28 +
             a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
             a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 * a_strx[(i + 2)-ifirst+1] -
             (a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
        a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm2_s25 +
        a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm2_s33 * a_strx[(i - 2)-ifirst+1])) +
       c1 * (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
         a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp1_s27 +
         a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
         a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 * a_strx[(i + 1)-ifirst+1] -
         (a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm1_s26 +
          a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm1_s34 * a_strx[(i - 1)-ifirst+1]))))
dudrm2_s25 = Float64(0)
dudrm1_s26 = Float64(0)
dudrp1_s27 = Float64(0)
dudrp2_s28 = Float64(0)
dvdrm2_s29 = Float64(0)
dvdrm1_s30 = Float64(0)
dvdrp1_s31 = Float64(0)
dvdrp2_s32 = Float64(0)
dwdrm2_s33 = Float64(0)
dwdrm1_s34 = Float64(0)
dwdrp1_s35 = Float64(0)
dwdrp2_s36 = Float64(0)
for q_s38 in Int32(1):Int32(8)
dudrm2_s25 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s38)+nijk*(1)+1])
dvdrm2_s29 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s38)+nijk*(2)+1])
dwdrm2_s33 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s38)+nijk*(3)+1])
dudrm1_s26 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s38)+nijk*(1)+1])
dvdrm1_s30 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s38)+nijk*(2)+1])
dwdrm1_s34 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s38)+nijk*(3)+1])
dudrp2_s28 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s38)+nijk*(1)+1])
dvdrp2_s32 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s38)+nijk*(2)+1])
dwdrp2_s36 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s38)+nijk*(3)+1])
dudrp1_s27 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s38)+nijk*(1)+1])
dvdrp1_s31 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s38)+nijk*(2)+1])
dwdrp1_s35 += (a_bope[(k)-1+6*(( q_s38)-1)+1] * a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s38)+nijk*(3)+1])
end
r1_s5 += (c2 * (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dudrp2_s28 * a_stry[(j + 2)-jfirst+1] * istrx_s3 +
          a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 -
          (a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dudrm2_s25 * a_stry[(j - 2)-jfirst+1] * istrx_s3 +
           a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dvdrm2_s29)) +
      c1 * (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dudrp1_s27 * a_stry[(j + 1)-jfirst+1] * istrx_s3 +
          a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 -
          (a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dudrm1_s26 * a_stry[(j - 1)-jfirst+1] * istrx_s3 +
           a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dvdrm1_s30)))
r2_s6 += (c2 * (a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dudrp2_s28 +
        (2 * a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1]) *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 *
        a_stry[(j + 2)-jfirst+1] * istrx_s3 +
        a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(4)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 * istrx_s3 -
        (a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dudrm2_s25 +
         (2 * a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1]) *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
         dvdrm2_s29 * a_stry[(j - 2)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(4)+1] *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dwdrm2_s33 * istrx_s3)) +
      c1 * (a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dudrp1_s27 +
          (2 * a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1]) *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 *
          a_stry[(j + 1)-jfirst+1] * istrx_s3 +
          a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 * istrx_s3 -
          (a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dudrm1_s26 +
           (2 * a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1]) *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
           dvdrm1_s30 * a_stry[(j - 1)-jfirst+1] * istrx_s3 +
           a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dwdrm1_s34 * istrx_s3)))
r3_s7 += ((c2 * (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 * a_stry[(j + 2)-jfirst+1] +
          a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 -
          (a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dwdrm2_s33 * a_stry[(j - 2)-jfirst+1] +
           a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dvdrm2_s29)) +
        c1 * (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 * a_stry[(j + 1)-jfirst+1] +
          a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 -
          (a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dwdrm1_s34 * a_stry[(j - 1)-jfirst+1] +
           a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dvdrm1_s30))) *
      istrx_s3)
for q_s39 in Int32(1):Int32(8)
r1_s5 += (a_bope[(k)-1+6*(( q_s39)-1)+1] *
        (
         
         (2 * a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] + a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2 +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1])) +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1])) *
         istry_s2
         
         + a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(1)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(2)+1]))))
r2_s6 += (a_bope[(k)-1+6*(( q_s39)-1)+1] *
        (
         
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1])) +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2
         
         + a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(1)+1])) +
         (2 * a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] + a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(2)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(3)+1])) *
         istrx_s3))
r3_s7 += (a_bope[(k)-1+6*(( q_s39)-1)+1] *
        (
         
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1])) *
         istry_s2 +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2
         
         + a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(3)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(2)+1])) *
         istrx_s3))
end
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] + sgn * r1_s5 * ijac_s1
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] + sgn * r2_s6 * ijac_s1
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] + sgn * r3_s7 * ijac_s1
end
return
end

function cpu_kernel2!(start0,N0,start1,N1,start2,N2,ifirst,ilast,jfirst,jlast,kfirst,klast,a1,sgn,a_u,a_mu,a_lambda,a_met,a_jac,a_lu,a_acof,a_bope,a_ghcof,a_acof_no_gp,a_ghcof_no_gp,a_strx,a_stry)
ni=ilast-ifirst+1
nij=ni*(jlast-jfirst+1)
nijk=nij*(klast-kfirst+1)
base=-(ifirst+ni*jfirst+nij*kfirst)
i6=1.0/6.0;tf=0.75;c1=2.0/3.0;c2=-1.0/12.0
i = Int32(start0)
j = Int32(start1)
k = Int32(start2)
if (i < N0) && (j < N1) && (k < N2)
ijac_s1 = Float64(a_strx[(i)-ifirst+1] * a_stry[(j)-jfirst+1] / a_jac[base+(i)+ni*( j)+nij*( k)+1])
istry_s2 = Float64(1 / (a_stry[(j)-jfirst+1]))
istrx_s3 = Float64(1 / (a_strx[(i)-ifirst+1]))
istrxy_s4 = Float64(istry_s2 * istrx_s3)
r1_s5 = Float64(0)
cof1_s6 = Float64((2 * a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 2)-ifirst+1])
cof2_s7 = Float64((2 * a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 1)-ifirst+1])
cof3_s8 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s9 = Float64((2 * a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 1)-ifirst+1])
cof5_s10 = Float64((2 * a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 2)-ifirst+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r1_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux2_s12 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux3_s13 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux4_s14 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
      istry_s2)
begin
cof1_s15 = Float64((a_mu[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s16 = Float64((a_mu[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s17 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s18 = Float64((a_mu[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s19 = Float64((a_mu[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s20 = Float64(cof2_s16 - tf * (cof3_s17 + cof1_s15))
mux2_s21 = Float64(cof1_s15 + cof4_s18 + 3 * (cof3_s17 + cof2_s16))
mux3_s22 = Float64(cof2_s16 + cof5_s19 + 3 * (cof4_s18 + cof3_s17))
mux4_s23 = Float64(cof4_s18 - tf * (cof3_s17 + cof5_s19))
r1_s5 += (i6 *
        (mux1_s20 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux2_s21 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux3_s22 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux4_s23 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
        istrx_s3)
end
begin
cof1_s24 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
        a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1]))
cof2_s25 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
        a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1]))
cof3_s26 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] *
        a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_mu[base+(i)+ni*( j)+nij*( k)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1]))
cof4_s27 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] *
        a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1]))
cof5_s28 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] *
        a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1]))
mux1_s29 = Float64(cof2_s25 - tf * (cof3_s26 + cof1_s24))
mux2_s30 = Float64(cof1_s24 + cof4_s27 + 3 * (cof3_s26 + cof2_s25))
mux3_s31 = Float64(cof2_s25 + cof5_s28 + 3 * (cof4_s27 + cof3_s26))
mux4_s32 = Float64(cof4_s27 - tf * (cof3_s26 + cof5_s28))
r1_s5 += (i6 *
        (mux1_s29 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux2_s30 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux3_s31 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux4_s32 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
        istrxy_s4)
end
cof1_s6 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1])
cof2_s7 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1])
cof3_s8 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])
cof4_s9 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1])
cof5_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r1_s5 += (i6 * (mux1_s11 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
        mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
        mux3_s13 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
        mux4_s14 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])))
cof1_s6 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1])
cof2_s7 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1])
cof3_s8 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1])
cof4_s9 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1])
cof5_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r1_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s13 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s14 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istry_s2)
r1_s5 += (c2 *
      (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1])) -
       a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]))) +
      c1 *
      (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1])) -
       a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))))
r1_s5 += (c2 *
      (a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1])) -
       a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))) +
      c1 *
      (a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1])) -
       a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))))
r1_s5 += (c2 *
      ((2 * a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) *
       a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1])) *
       a_strx[(i)-ifirst+1] * istry_s2 +
       a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1])) +
       a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1])) *
       istry_s2 -
       ((2 * a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1])) *
        a_strx[(i)-ifirst+1] * istry_s2 +
        a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1])) +
        a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1])) *
        istry_s2)) +
        c1 *
        ((2 * a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) *
         a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2 +
         a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1])) +
         a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1])) *
         istry_s2 -
         ((2 * a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1]) +
           c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) *
          a_strx[(i)-ifirst+1] * istry_s2 +
          a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1]) +
           c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) +
          a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1]) +
           c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) *
          istry_s2)))
r1_s5 += ((c2 *
       ((2 * a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1]) *
        a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) *
        a_strx[(i + 2)-ifirst+1] +
        a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
        a_stry[(j)-jfirst+1] +
        a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
        a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
         c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) -
        ((2 * a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1]) *
         a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
          c1 *
          (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) *
         a_strx[(i - 2)-ifirst+1] +
         a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
         a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
          c1 *
          (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
         a_stry[(j)-jfirst+1] +
         a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
         a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1] -
            a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1])))) +
            c1 *
            ((2 * a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1]) *
             a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
             (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
              c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) *
             a_strx[(i + 1)-ifirst+1] +
             a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
             a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
             (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
              c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
             a_stry[(j)-jfirst+1] +
             a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
             a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
             (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
              c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) -
             ((2 * a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1]) *
              a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
              (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
               c1 *
               (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) *
              a_strx[(i - 1)-ifirst+1] +
              a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
              a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
              (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
               c1 *
               (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
              a_stry[(j)-jfirst+1] +
              a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
              a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
              (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
               c1 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1] -
                 a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1]))))) *
                 istry_s2)
r1_s5 += (c2 *
      (a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(1)+1])) *
       a_stry[(j)-jfirst+1] * istrx_s3 +
       a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(2)+1])) -
       (a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(1)+1])) *
        a_stry[(j)-jfirst+1] * istrx_s3 +
        a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(2)+1] -
           a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(2)+1])))) +
      c1 *
      (a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(1)+1])) *
       a_stry[(j)-jfirst+1] * istrx_s3 +
       a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(2)+1])) -
       (a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(1)+1])) *
        a_stry[(j)-jfirst+1] * istrx_s3 +
        a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(2)+1]) +
         c1 *
         (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(2)+1])))))
r1_s5 += (c2 *
      (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(1)+1])) *
       a_stry[(j + 2)-jfirst+1] * istrx_s3 +
       a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(2)+1])) -
       (a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(1)+1])) *
        a_stry[(j - 2)-jfirst+1] * istrx_s3 +
        a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(2)+1] -
           a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(2)+1])))) +
      c1 *
      (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(1)+1])) *
       a_stry[(j + 1)-jfirst+1] * istrx_s3 +
       a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(2)+1])) -
       (a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(1)+1])) *
        a_stry[(j - 1)-jfirst+1] * istrx_s3 +
        a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(2)+1]) +
         c1 *
         (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(2)+1])))))
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] + sgn * r1_s5 * ijac_s1
end
return
end

function cpu_kernel3!(start0,N0,start1,N1,start2,N2,ifirst,ilast,jfirst,jlast,kfirst,klast,a1,sgn,a_u,a_mu,a_lambda,a_met,a_jac,a_lu,a_acof,a_bope,a_ghcof,a_acof_no_gp,a_ghcof_no_gp,a_strx,a_stry)
ni=ilast-ifirst+1
nij=ni*(jlast-jfirst+1)
nijk=nij*(klast-kfirst+1)
base=-(ifirst+ni*jfirst+nij*kfirst)
i6=1.0/6.0;tf=0.75;c1=2.0/3.0;c2=-1.0/12.0
i = Int32(start0)
j = Int32(start1)
k = Int32(start2)
if (i < N0) && (j < N1) && (k < N2)
ijac_s1 = Float64(a_strx[(i)-ifirst+1] * a_stry[(j)-jfirst+1] / a_jac[base+(i)+ni*( j)+nij*( k)+1])
istry_s2 = Float64(1 / (a_stry[(j)-jfirst+1]))
istrx_s3 = Float64(1 / (a_strx[(i)-ifirst+1]))
istrxy_s4 = Float64(istry_s2 * istrx_s3)
r2_s5 = Float64(0)
cof1_s6 = Float64((a_mu[base+(i - 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 2)-ifirst+1])
cof2_s7 = Float64((a_mu[base+(i - 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 1)-ifirst+1])
cof3_s8 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s9 = Float64((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 1)-ifirst+1])
cof5_s10 = Float64((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 2)-ifirst+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r2_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s12 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s13 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s14 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istry_s2)
cof1_s6 = Float64((2 * a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s7 = Float64((2 * a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s8 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s9 = Float64((2 * a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s10 = Float64((2 * a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r2_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s12 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s13 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s14 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istrx_s3)
cof1_s6 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1])
cof2_s7 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1])
cof3_s8 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])
cof4_s9 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1])
cof5_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r2_s5 += (i6 * (mux1_s11 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
        mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
        mux3_s13 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
        mux4_s14 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])))
cof1_s6 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
      a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
      a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1]))
cof2_s7 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
      a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
      a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1]))
cof3_s8 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
      a_mu[base+(i)+ni*( j)+nij*( k)+1] *
      (a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
       a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1]))
cof4_s9 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] *
      a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
      a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
          a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1]))
cof5_s10 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] *
      a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
      a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
          a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1]))
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r2_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s13 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s14 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istrxy_s4)
cof1_s6 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1])
cof2_s7 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1])
cof3_s8 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1])
cof4_s9 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1])
cof5_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r2_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s13 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s14 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istrx_s3)
r2_s5 += (c2 *
      (a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1])) -
       a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]))) +
      c1 *
      (a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1])) -
       a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))))
r2_s5 += (c2 *
      (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1])) -
       a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))) +
      c1 *
      (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1])) -
       a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))))
r2_s5 += (c2 *
      ((a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] *
       a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1])) +
       a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1])) *
       a_strx[(i)-ifirst+1] * istry_s2 -
       ((a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1])) +
        a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1])) *
        a_strx[(i)-ifirst+1] * istry_s2)) +
      c1 *
      ((a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] *
       a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1])) +
       a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1])) *
       a_strx[(i)-ifirst+1] * istry_s2 -
       (a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
        a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
        a_strx[(i)-ifirst+1] * istry_s2)))
r2_s5 += (c2 *
      ((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
       a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
       a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
       a_strx[(i + 2)-ifirst+1] * istry_s2 -
       (a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
        a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
        a_strx[(i - 2)-ifirst+1] * istry_s2)) +
      c1 *
      ((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
       a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
       a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
       a_strx[(i + 1)-ifirst+1] * istry_s2 -
       (a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
        a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(2)+1])) *
        a_strx[(i - 1)-ifirst+1] * istry_s2)))
r2_s5 += (c2 *
      (a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(1)+1])) +
       (2 * a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) *
       a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(2)+1])) *
       a_stry[(j)-jfirst+1] * istrx_s3 +
       a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(3)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(3)+1])) *
       istrx_s3 -
       (a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(1)+1])) +
        (2 * a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(2)+1])) *
        a_stry[(j)-jfirst+1] * istrx_s3 +
        a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(3)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(3)+1])) *
        istrx_s3)) +
        c1 *
        (a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(1)+1])) +
         (2 * a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) *
         a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(2)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(3)+1])) *
         istrx_s3 -
         (a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(1)+1]) +
           c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(1)+1])) +
          (2 * a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(2)+1]) +
           c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(2)+1])) *
          a_stry[(j)-jfirst+1] * istrx_s3 +
          a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(3)+1]) +
           c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(3)+1])) *
          istrx_s3)))
r2_s5 += (c2 *
      (a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(1)+1])) +
       (2 * a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1]) *
       a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(2)+1])) *
       a_stry[(j + 2)-jfirst+1] * istrx_s3 +
       a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(3)+1]) +
        c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(3)+1])) *
       istrx_s3 -
       (a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(1)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(1)+1])) +
        (2 * a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1]) *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(2)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(2)+1])) *
        a_stry[(j - 2)-jfirst+1] * istrx_s3 +
        a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(4)+1] *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(3)+1]) +
         c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(3)+1])) *
        istrx_s3)) +
        c1 *
        (a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(1)+1])) +
         (2 * a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1]) *
         a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(2)+1])) *
         a_stry[(j + 1)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(3)+1])) *
         istrx_s3 -
         (a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(1)+1]) +
           c1 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(1)+1])) +
          (2 * a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1]) *
          a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(2)+1]) +
           c1 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(2)+1])) *
          a_stry[(j - 1)-jfirst+1] * istrx_s3 +
          a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
          (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(3)+1]) +
           c1 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(3)+1])) *
          istrx_s3)))
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] + sgn * r2_s5 * ijac_s1
end
return
end

function cpu_kernel4!(start0,N0,start1,N1,start2,N2,ifirst,ilast,jfirst,jlast,kfirst,klast,a1,sgn,a_u,a_mu,a_lambda,a_met,a_jac,a_lu,a_acof,a_bope,a_ghcof,a_acof_no_gp,a_ghcof_no_gp,a_strx,a_stry)
ni=ilast-ifirst+1
nij=ni*(jlast-jfirst+1)
nijk=nij*(klast-kfirst+1)
base=-(ifirst+ni*jfirst+nij*kfirst)
i6=1.0/6.0;tf=0.75;c1=2.0/3.0;c2=-1.0/12.0
i = Int32(start0)
j = Int32(start1)
k = Int32(start2)
if (i < N0) && (j < N1) && (k < N2)
ijac_s1 = Float64(a_strx[(i)-ifirst+1] * a_stry[(j)-jfirst+1] / a_jac[base+(i)+ni*( j)+nij*( k)+1])
istry_s2 = Float64(1 / (a_stry[(j)-jfirst+1]))
istrx_s3 = Float64(1 / (a_strx[(i)-ifirst+1]))
istrxy_s4 = Float64(istry_s2 * istrx_s3)
r3_s5 = Float64(0.0)
cof1_s6 = Float64((a_mu[base+(i - 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 2)-ifirst+1])
cof2_s7 = Float64((a_mu[base+(i - 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i - 1)-ifirst+1])
cof3_s8 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s9 = Float64((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 1)-ifirst+1])
cof5_s10 = Float64((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i + 2)-ifirst+1])
mux1_s11 = Float64(cof2_s7 - tf * (cof3_s8 + cof1_s6))
mux2_s12 = Float64(cof1_s6 + cof4_s9 + 3 * (cof3_s8 + cof2_s7))
mux3_s13 = Float64(cof2_s7 + cof5_s10 + 3 * (cof4_s9 + cof3_s8))
mux4_s14 = Float64(cof4_s9 - tf * (cof3_s8 + cof5_s10))
r3_s5 += (i6 *
      (mux1_s11 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s12 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s13 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s14 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istry_s2)
begin
local cof1_s15
local cof2_s16
local cof3_s17
local cof4_s18
local cof5_s19
local mux1_s20
local mux3_s21
local mux4_s22
cof1_s15 = Float64((a_mu[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s16 = Float64((a_mu[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s17 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s18 = Float64((a_mu[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s19 = Float64((a_mu[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s20 = Float64(cof2_s16 - tf * (cof3_s17 + cof1_s15))
mux2_s12 = Float64(cof1_s15 + cof4_s18 + 3 * (cof3_s17 + cof2_s16))
mux3_s21 = Float64(cof2_s16 + cof5_s19 + 3 * (cof4_s18 + cof3_s17))
mux4_s22 = Float64(cof4_s18 - tf * (cof3_s17 + cof5_s19))
r3_s5 += (i6 *
        (mux1_s20 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
         mux2_s12 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
         mux3_s21 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
         mux4_s22 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
        istrx_s3)
end
begin
local cof1_s23
local cof2_s24
local cof3_s25
local cof4_s26
local cof5_s27
local mux1_s28
local mux3_s29
local mux4_s30
cof1_s23 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1])
cof2_s24 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1])
cof3_s25 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1])
cof4_s26 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1])
cof5_s27 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1])
mux1_s28 = Float64(cof2_s24 - tf * (cof3_s25 + cof1_s23))
mux2_s12 = Float64(cof1_s23 + cof4_s26 + 3 * (cof3_s25 + cof2_s24))
mux3_s29 = Float64(cof2_s24 + cof5_s27 + 3 * (cof4_s26 + cof3_s25))
mux4_s30 = Float64(cof4_s26 - tf * (cof3_s25 + cof5_s27))
r3_s5 += (i6 *
        (mux1_s28 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux3_s29 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
         mux4_s30 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
        istry_s2)
end
begin
local cof1_s31
local cof2_s32
local cof3_s33
local cof4_s34
local cof5_s35
local mux1_s36
local mux3_s37
local mux4_s38
cof1_s31 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1])
cof2_s32 = Float64((a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1])
cof3_s33 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1])
cof4_s34 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1])
cof5_s35 = Float64((a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1])
mux1_s36 = Float64(cof2_s32 - tf * (cof3_s33 + cof1_s31))
mux2_s12 = Float64(cof1_s31 + cof4_s34 + 3 * (cof3_s33 + cof2_s32))
mux3_s37 = Float64(cof2_s32 + cof5_s35 + 3 * (cof4_s34 + cof3_s33))
mux4_s38 = Float64(cof4_s34 - tf * (cof3_s33 + cof5_s35))
r3_s5 += (i6 *
        (mux1_s36 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
         mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
         mux3_s37 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
         mux4_s38 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
        istrx_s3)
end
begin
local cof1_s39
local cof2_s40
local cof3_s41
local cof4_s42
local cof5_s43
local mux1_s44
local mux3_s45
local mux4_s46
cof1_s39 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] +
        a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1]))
cof2_s40 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1]) *
        a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] +
        a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1]))
cof3_s41 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1] *
        a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(4)+1] +
        a_mu[base+(i)+ni*( j)+nij*( k)+1] *
        (a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
         a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] * a_stry[(j)-jfirst+1]))
cof4_s42 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) *
        a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] +
        a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] * a_stry[(j)-jfirst+1]))
cof5_s43 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] + a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) *
        a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] +
        a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] * a_stry[(j)-jfirst+1]))
mux1_s44 = Float64(cof2_s40 - tf * (cof3_s41 + cof1_s39))
mux2_s12 = Float64(cof1_s39 + cof4_s42 + 3 * (cof3_s41 + cof2_s40))
mux3_s45 = Float64(cof2_s40 + cof5_s43 + 3 * (cof4_s42 + cof3_s41))
mux4_s46 = Float64(cof4_s42 - tf * (cof3_s41 + cof5_s43))
r3_s5 += (i6 *
        (mux1_s44 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
         mux2_s12 * (a_u[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
         mux3_s45 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
         mux4_s46 * (a_u[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
        istrxy_s4
        
        
        
        +
        c2 *
        ((a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1]) +
          c1 *
          (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1])) *
         istry_s2 +
         a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(2)+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1]) +
          c1 *
          (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2 -
         ((a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
          (c2 *
           (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
           c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1] -
             a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1])) *
          istry_s2 +
          a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
          (c2 *
           (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
           c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1] -
             a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1])) *
          a_strx[(i)-ifirst+1] * istry_s2)) +
          c1 *
          ((a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
           (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1]) +
            c1 *
            (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1])) *
           istry_s2 +
           a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(2)+1] *
           a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
           (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1]) +
            c1 *
            (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1])) *
           a_strx[(i)-ifirst+1] * istry_s2 -
           (a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
            (c2 *
             (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1]) +
             c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1] -
               a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) *
            istry_s2 +
            a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(2)+1] *
            a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
            (c2 *
             (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1]) +
             c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1] -
               a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) *
            a_strx[(i)-ifirst+1] * istry_s2))
            
            
            
            + istry_s2 * (c2 * ((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
                  a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
                  (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] -
                   a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
                   c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] -
                     a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
                  a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
                  a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
                  (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1] -
                   a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
                   c1 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1] -
                     a_u[base-nijk+( i + 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) *
                  a_strx[(i + 2)-ifirst+1] -
                  (a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
                   a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
                   (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(1)+1] -
                    a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
                    c1 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(1)+1] -
                      a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
                   a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
                   a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
                   (c2 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 2)+nijk*(3)+1] -
                    a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
                    c1 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k + 1)+nijk*(3)+1] -
                      a_u[base-nijk+( i - 2)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) *
                   a_strx[(i - 2)-ifirst+1])) +
                   c1 * ((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
                       a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
                       (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] -
                        a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
                        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] -
                          a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
                       a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
                       a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
                       (c2 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1] -
                        a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
                        c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1] -
                          a_u[base-nijk+( i + 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) *
                       a_strx[(i + 1)-ifirst+1] -
                       (a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
                        a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
                        (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(1)+1] -
                         a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(1)+1]) +
                         c1 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(1)+1] -
                           a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(1)+1])) +
                        a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
                        a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
                        (c2 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 2)+nijk*(3)+1] -
                         a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 2)+nijk*(3)+1]) +
                         c1 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k + 1)+nijk*(3)+1] -
                           a_u[base-nijk+( i - 1)+ni*( j)+nij*( k - 1)+nijk*(3)+1])) *
                        a_strx[(i - 1)-ifirst+1])))
                        
                        
                        
                        +
                        c2 *
                        (a_mu[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(3)+1] *
                         a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
                         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(3)+1]) +
                    c1 *
                    (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(3)+1])) *
                         a_stry[(j)-jfirst+1] * istrx_s3 +
                         a_lambda[base+(i)+ni*( j)+nij*( k + 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(4)+1] *
                         a_met[base-nijk+( i)+ni*( j)+nij*( k + 2)+nijk*(1)+1] *
                         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(2)+1]) +
                    c1 *
                    (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(2)+1])) *
                         istrx_s3 -
                         (a_mu[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(3)+1] *
                    a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
                    (c2 *
                     (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(3)+1]) +
                     c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(3)+1] -
                       a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(3)+1])) *
                    a_stry[(j)-jfirst+1] * istrx_s3 +
                    a_lambda[base+(i)+ni*( j)+nij*( k - 2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(4)+1] *
                    a_met[base-nijk+( i)+ni*( j)+nij*( k - 2)+nijk*(1)+1] *
                    (c2 *
                     (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(2)+1]) +
                     c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(2)+1] -
                       a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(2)+1])) *
                    istrx_s3)) +
                    c1 *
                    (a_mu[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(3)+1] *
                     a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
                     (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(3)+1]) +
                      c1 *
                      (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(3)+1])) *
                     a_stry[(j)-jfirst+1] * istrx_s3 +
                     a_lambda[base+(i)+ni*( j)+nij*( k + 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(4)+1] *
                     a_met[base-nijk+( i)+ni*( j)+nij*( k + 1)+nijk*(1)+1] *
                     (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(2)+1]) +
                      c1 *
                      (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(2)+1])) *
                     istrx_s3 -
                     (a_mu[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(3)+1] *
                      a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
                      (c2 *
                       (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(3)+1]) +
                       c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(3)+1] -
                         a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(3)+1])) *
                      a_stry[(j)-jfirst+1] * istrx_s3 +
                      a_lambda[base+(i)+ni*( j)+nij*( k - 1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(4)+1] *
                      a_met[base-nijk+( i)+ni*( j)+nij*( k - 1)+nijk*(1)+1] *
                      (c2 *
                       (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(2)+1]) +
                       c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(2)+1] -
                         a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(2)+1])) *
                      istrx_s3))
                      
                      
                      
                      + istrx_s3 * (c2 * (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] *
                            a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
                            (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(3)+1] -
                             a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(3)+1]) +
                             c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(3)+1] -
                               a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(3)+1])) *
                            a_stry[(j + 2)-jfirst+1] +
                            a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(4)+1] *
                            a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
                            (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 2)+nijk*(2)+1] -
                             a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 2)+nijk*(2)+1]) +
                             c1 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k + 1)+nijk*(2)+1] -
                               a_u[base-nijk+( i)+ni*( j + 2)+nij*( k - 1)+nijk*(2)+1])) -
                            (a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] *
                             a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
                             (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(3)+1] -
                              a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(3)+1]) +
                              c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(3)+1] -
                                a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(3)+1])) *
                             a_stry[(j - 2)-jfirst+1] +
                             a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(4)+1] *
                             a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
                             (c2 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 2)+nijk*(2)+1] -
                              a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 2)+nijk*(2)+1]) +
                              c1 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k + 1)+nijk*(2)+1] -
                                a_u[base-nijk+( i)+ni*( j - 2)+nij*( k - 1)+nijk*(2)+1])))) +
                                c1 * (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] *
                                    a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
                                    (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(3)+1] -
                                     a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(3)+1]) +
                                     c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(3)+1] -
                                       a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(3)+1])) *
                                    a_stry[(j + 1)-jfirst+1] +
                                    a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(4)+1] *
                                    a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
                                    (c2 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 2)+nijk*(2)+1] -
                                     a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 2)+nijk*(2)+1]) +
                                     c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k + 1)+nijk*(2)+1] -
                                       a_u[base-nijk+( i)+ni*( j + 1)+nij*( k - 1)+nijk*(2)+1])) -
                                    (a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] *
                                     a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
                                     (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(3)+1] -
                                      a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(3)+1]) +
                                      c1 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(3)+1] -
                                        a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(3)+1])) *
                                     a_stry[(j - 1)-jfirst+1] +
                                     a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(4)+1] *
                                     a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
                                     (c2 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 2)+nijk*(2)+1] -
                                      a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 2)+nijk*(2)+1]) +
                                      c1 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k + 1)+nijk*(2)+1] -
                                        a_u[base-nijk+( i)+ni*( j - 1)+nij*( k - 1)+nijk*(2)+1]))))))
end
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] + sgn * r3_s5 * ijac_s1
end
return
end

function cpu_kernel5!(start0,N0,start1,N1,start2,N2,ifirst,ilast,jfirst,jlast,kfirst,klast,nk,a1,sgn,a_u,a_mu,a_lambda,a_met,a_jac,a_lu,a_acof,a_bope,a_ghcof,a_acof_no_gp,a_ghcof_no_gp,a_strx,a_stry)
ni=ilast-ifirst+1
nij=ni*(jlast-jfirst+1)
nijk=nij*(klast-kfirst+1)
base=-(ifirst+ni*jfirst+nij*kfirst)
i6=1.0/6.0;tf=0.75;c1=2.0/3.0;c2=-1.0/12.0
i = Int32(start0)
j = Int32(start1)
k = Int32(start2)
if (i < N0) && (j < N1) && (k < N2)
ijac_s1 = Float64(a_strx[(i)-ifirst+1] * a_stry[(j)-jfirst+1] / a_jac[base+(i)+ni*( j)+nij*( k)+1])
istry_s2 = Float64(1 / (a_stry[(j)-jfirst+1]))
istrx_s3 = Float64(1 / (a_strx[(i)-ifirst+1]))
istrxy_s4 = Float64(istry_s2 * istrx_s3)
r1_s5 = Float64(0)
r2_s6 = Float64(0)
r3_s7 = Float64(0)
cof1_s8 = Float64((2 * a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 2)-ifirst+1])
cof2_s9 = Float64((2 * a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 1)-ifirst+1])
cof3_s10 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s11 = Float64((2 * a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 1)-ifirst+1])
cof5_s12 = Float64((2 * a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1]) *
      a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 2)-ifirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r1_s5 = Float64(r1_s5 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux2_s14 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux3_s15 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux4_s16 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
      istry_s2)
cof1_s8 = Float64((a_mu[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
      a_stry[(j - 2)-jfirst+1])
cof2_s9 = Float64((a_mu[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
      a_stry[(j - 1)-jfirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s11 = Float64((a_mu[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
      a_stry[(j + 1)-jfirst+1])
cof5_s12 = Float64((a_mu[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
      a_stry[(j + 2)-jfirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r1_s5 = Float64(r1_s5 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux2_s14 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux3_s15 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1]) +
       mux4_s16 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1])) *
      istrx_s3)
cof1_s8 = Float64((a_mu[base+(i - 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 2)-ifirst+1])
cof2_s9 = Float64((a_mu[base+(i - 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 1)-ifirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s11 = Float64((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 1)-ifirst+1])
cof5_s12 = Float64((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 2)-ifirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r2_s6 = Float64(r2_s6 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s14 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s15 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s16 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istry_s2)
cof1_s8 = Float64((2 * a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_stry[(j - 2)-jfirst+1])
cof2_s9 = Float64((2 * a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_stry[(j - 1)-jfirst+1])
cof3_s10 = Float64((2 * a_mu[base+(i)+ni*( j)+nij*( k)+1] + a_lambda[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s11 = Float64((2 * a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_stry[(j + 1)-jfirst+1])
cof5_s12 = Float64((2 * a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
      a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_stry[(j + 2)-jfirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r2_s6 = Float64(r2_s6 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux2_s14 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux3_s15 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1]) +
       mux4_s16 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1])) *
      istrx_s3)
cof1_s8 = Float64((a_mu[base+(i - 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 2)-ifirst+1])
cof2_s9 = Float64((a_mu[base+(i - 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i - 1)-ifirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_strx[(i)-ifirst+1])
cof4_s11 = Float64((a_mu[base+(i + 1)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 1)-ifirst+1])
cof5_s12 = Float64((a_mu[base+(i + 2)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
      a_strx[(i + 2)-ifirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r3_s7 = Float64(r3_s7 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s14 * (a_u[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s15 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s16 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istry_s2)
cof1_s8 = Float64((a_mu[base+(i)+ni*( j - 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
      a_stry[(j - 2)-jfirst+1])
cof2_s9 = Float64((a_mu[base+(i)+ni*( j - 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
      a_stry[(j - 1)-jfirst+1])
cof3_s10 = Float64((a_mu[base+(i)+ni*( j)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] * a_stry[(j)-jfirst+1])
cof4_s11 = Float64((a_mu[base+(i)+ni*( j + 1)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
      a_stry[(j + 1)-jfirst+1])
cof5_s12 = Float64((a_mu[base+(i)+ni*( j + 2)+nij*( k)+1]) * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
      a_stry[(j + 2)-jfirst+1])
mux1_s13 = Float64(cof2_s9 - tf * (cof3_s10 + cof1_s8))
mux2_s14 = Float64(cof1_s8 + cof4_s11 + 3 * (cof3_s10 + cof2_s9))
mux3_s15 = Float64(cof2_s9 + cof5_s12 + 3 * (cof4_s11 + cof3_s10))
mux4_s16 = Float64(cof4_s11 - tf * (cof3_s10 + cof5_s12))
r3_s7 = Float64(r3_s7 + i6 *
      (mux1_s13 * (a_u[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux2_s14 * (a_u[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux3_s15 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1]) +
       mux4_s16 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1])) *
      istrx_s3)
local mucofu2_s17
local mucofuv_s18
local mucofuw_s19
local mucofvw_s20
local mucofv2_s21
local mucofw2_s22
for q_s23 in Int32(nk - 7):Int32(nk)
mucofu2_s17 = Float64(0)
mucofuv_s18 = Float64(0)
mucofuw_s19 = Float64(0)
mucofvw_s20 = Float64(0)
mucofv2_s21 = Float64(0)
mucofw2_s22 = Float64(0)
for m_s24 in Int32(nk - 7):Int32(nk)
mucofu2_s17 += (a_acof_no_gp[(nk - k + 1)-1+6*(( nk - q_s23 + 1)-1)+48*(( nk - m_s24 + 1)-1)+1] *
          ((2 * a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] *
           a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
           a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])))
mucofv2_s21 += (a_acof_no_gp[(nk - k + 1)-1+6*(( nk - q_s23 + 1)-1)+48*(( nk - m_s24 + 1)-1)+1] *
          ((2 * a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] *
           a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
           a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
             a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])))
mucofw2_s22 += (a_acof_no_gp[(nk - k + 1)-1+6*(( nk - q_s23 + 1)-1)+48*(( nk - m_s24 + 1)-1)+1] *
          ((2 * a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1] +
           a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] *
           (a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
            a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] * a_stry[(j)-jfirst+1])))
mucofuv_s18 += (a_acof_no_gp[(nk - k + 1)-1+6*(( nk - q_s23 + 1)-1)+48*(( nk - m_s24 + 1)-1)+1] *
          (a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1])
mucofuw_s19 += (a_acof_no_gp[(nk - k + 1)-1+6*(( nk - q_s23 + 1)-1)+48*(( nk - m_s24 + 1)-1)+1] *
          (a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])
mucofvw_s20 += (a_acof_no_gp[(nk - k + 1)-1+6*(( nk - q_s23 + 1)-1)+48*(( nk - m_s24 + 1)-1)+1] *
          (a_mu[base+(i)+ni*( j)+nij*( m_s24)+1] + a_lambda[base+(i)+ni*( j)+nij*( m_s24)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j)+nij*( m_s24)+nijk*(4)+1])
end
r1_s5 += (istrxy_s4 * mucofu2_s17 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(1)+1] + mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(2)+1] +
        istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(3)+1])
r2_s6 += (mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(1)+1] + istrxy_s4 * mucofv2_s21 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(2)+1] +
        istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(3)+1])
r3_s7 += (istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(1)+1] +
        istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(2)+1] +
        istrxy_s4 * mucofw2_s22 * a_u[base-nijk+( i)+ni*( j)+nij*( q_s23)+nijk*(3)+1])
end
mucofu2_s17 = Float64(a_ghcof_no_gp[(nk - k + 1)] *
      ((2 * a_mu[base+(i)+ni*( j)+nij*( nk)+1] + a_lambda[base+(i)+ni*( j)+nij*( nk)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] *
       a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
       a_mu[base+(i)+ni*( j)+nij*( nk)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] * a_stry[(j)-jfirst+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
         a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1])))
mucofv2_s21 = Float64(a_ghcof_no_gp[(nk - k + 1)] *
      ((2 * a_mu[base+(i)+ni*( j)+nij*( nk)+1] + a_lambda[base+(i)+ni*( j)+nij*( nk)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] *
       a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] * a_stry[(j)-jfirst+1] +
       a_mu[base+(i)+ni*( j)+nij*( nk)+1] * (a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_strx[(i)-ifirst+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
         a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1])))
mucofw2_s22 = Float64(a_ghcof_no_gp[(nk - k + 1)] *
      ((2 * a_mu[base+(i)+ni*( j)+nij*( nk)+1] + a_lambda[base+(i)+ni*( j)+nij*( nk)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1] *
       a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1] +
       a_mu[base+(i)+ni*( j)+nij*( nk)+1] *
       (a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_strx[(i)-ifirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_strx[(i)-ifirst+1] +
        a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] * a_stry[(j)-jfirst+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] * a_stry[(j)-jfirst+1])))
mucofuv_s18 = Float64(a_ghcof_no_gp[(nk - k + 1)] * (a_mu[base+(i)+ni*( j)+nij*( nk)+1] + a_lambda[base+(i)+ni*( j)+nij*( nk)+1]) *
      a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1])
mucofuw_s19 = Float64(a_ghcof_no_gp[(nk - k + 1)] * (a_mu[base+(i)+ni*( j)+nij*( nk)+1] + a_lambda[base+(i)+ni*( j)+nij*( nk)+1]) *
      a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1])
mucofvw_s20 = Float64(a_ghcof_no_gp[(nk - k + 1)] * (a_mu[base+(i)+ni*( j)+nij*( nk)+1] + a_lambda[base+(i)+ni*( j)+nij*( nk)+1]) *
      a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( nk)+nijk*(4)+1])
r1_s5 += (istrxy_s4 * mucofu2_s17 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(1)+1] +
      mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(2)+1] +
      istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(3)+1])
r2_s6 += (mucofuv_s18 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(1)+1] +
      istrxy_s4 * mucofv2_s21 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(2)+1] +
      istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(3)+1])
r3_s7 += (istry_s2 * mucofuw_s19 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(1)+1] +
      istrx_s3 * mucofvw_s20 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(2)+1] +
      istrxy_s4 * mucofw2_s22 * a_u[base-nijk+( i)+ni*( j)+nij*( nk + 1)+nijk*(3)+1])
r1_s5 += (c2 *
      (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1])) -
       a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]))) +
      c1 *
      (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1])) -
       a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))))
r1_s5 += (c2 *
      (a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1])) -
       a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))) +
      c1 *
      (a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1])) -
       a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(2)+1]) +
        c1 * (a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(2)+1]))))
r2_s6 += (c2 *
      (a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1])) -
       a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]))) +
      c1 *
      (a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1])) -
       a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))))
r2_s6 += (c2 *
      (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1])) -
       a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 2)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i - 2)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))) +
      c1 *
      (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i + 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i + 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i + 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1])) -
       a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
       (c2 * (a_u[base-nijk+( i - 1)+ni*( j + 2)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 2)+nij*( k)+nijk*(1)+1]) +
        c1 * (a_u[base-nijk+( i - 1)+ni*( j + 1)+nij*( k)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j - 1)+nij*( k)+nijk*(1)+1]))))
dudrm2_s25 = Float64(0)
dudrm1_s26 = Float64(0)
dudrp1_s27 = Float64(0)
dudrp2_s28 = Float64(0)
dvdrm2_s29 = Float64(0)
dvdrm1_s30 = Float64(0)
dvdrp1_s31 = Float64(0)
dvdrp2_s32 = Float64(0)
dwdrm2_s33 = Float64(0)
dwdrm1_s34 = Float64(0)
dwdrp1_s35 = Float64(0)
dwdrp2_s36 = Float64(0)
for q_s37 in Int32(nk - 7):Int32(nk)
dudrm2_s25 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrm2_s29 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrm2_s33 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
dudrm1_s26 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrm1_s30 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrm1_s34 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
dudrp2_s28 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrp2_s32 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrp2_s36 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
dudrp1_s27 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s37)+nijk*(1)+1])
dvdrp1_s31 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s37)+nijk*(2)+1])
dwdrp1_s35 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s37 + 1)-1)+1] * a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s37)+nijk*(3)+1])
end
r1_s5 += ((c2 * ((2 * a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1]) *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
          a_strx[(i + 2)-ifirst+1] * dudrp2_s28 +
          a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 * a_stry[(j)-jfirst+1] +
          a_lambda[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 -
          ((2 * a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1]) *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] *
           a_strx[(i - 2)-ifirst+1] * dudrm2_s25 +
           a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm2_s29 * a_stry[(j)-jfirst+1] +
           a_lambda[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm2_s33)) +
        c1 * ((2 * a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1]) *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
          a_strx[(i + 1)-ifirst+1] * dudrp1_s27 +
          a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 * a_stry[(j)-jfirst+1] +
          a_lambda[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 -
          ((2 * a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] + a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1]) *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] *
           a_strx[(i - 1)-ifirst+1] * dudrm1_s26 +
           a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm1_s30 * a_stry[(j)-jfirst+1] +
           a_lambda[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm1_s34))) *
           istry_s2)
r2_s6 += (c2 * (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp2_s28 +
          a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 * a_strx[(i + 2)-ifirst+1] * istry_s2 -
          (a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm2_s25 +
           a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm2_s29 * a_strx[(i - 2)-ifirst+1] * istry_s2)) +
      c1 * (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp1_s27 +
          a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 * a_strx[(i + 1)-ifirst+1] * istry_s2 -
          (a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm1_s26 +
           a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dvdrm1_s30 * a_strx[(i - 1)-ifirst+1] * istry_s2)))
r3_s7 += (istry_s2 * (c2 * (a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
            a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp2_s28 +
            a_mu[base+(i + 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
            a_met[base-nijk+( i + 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 * a_strx[(i + 2)-ifirst+1] -
            (a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(4)+1] *
             a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm2_s25 +
             a_mu[base+(i - 2)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(2)+1] *
             a_met[base-nijk+( i - 2)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm2_s33 * a_strx[(i - 2)-ifirst+1])) +
          c1 * (a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
            a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrp1_s27 +
            a_mu[base+(i + 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
            a_met[base-nijk+( i + 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 * a_strx[(i + 1)-ifirst+1] -
            (a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(4)+1] *
             a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dudrm1_s26 +
             a_mu[base+(i - 1)+ni*( j)+nij*( k)+1] * a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(2)+1] *
             a_met[base-nijk+( i - 1)+ni*( j)+nij*( k)+nijk*(1)+1] * dwdrm1_s34 * a_strx[(i - 1)-ifirst+1]))))
dudrm2_s25 = Float64(0)
dudrm1_s26 = Float64(0)
dudrp1_s27 = Float64(0)
dudrp2_s28 = Float64(0)
dvdrm2_s29 = Float64(0)
dvdrm1_s30 = Float64(0)
dvdrp1_s31 = Float64(0)
dvdrp2_s32 = Float64(0)
dwdrm2_s33 = Float64(0)
dwdrm1_s34 = Float64(0)
dwdrp1_s35 = Float64(0)
dwdrp2_s36 = Float64(0)
for q_s38 in Int32(nk - 7):Int32(nk)
dudrm2_s25 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s38)+nijk*(1)+1])
dvdrm2_s29 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s38)+nijk*(2)+1])
dwdrm2_s33 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s38)+nijk*(3)+1])
dudrm1_s26 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s38)+nijk*(1)+1])
dvdrm1_s30 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s38)+nijk*(2)+1])
dwdrm1_s34 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s38)+nijk*(3)+1])
dudrp2_s28 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s38)+nijk*(1)+1])
dvdrp2_s32 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s38)+nijk*(2)+1])
dwdrp2_s36 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s38)+nijk*(3)+1])
dudrp1_s27 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s38)+nijk*(1)+1])
dvdrp1_s31 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s38)+nijk*(2)+1])
dwdrp1_s35 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s38 + 1)-1)+1] * a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s38)+nijk*(3)+1])
end
r1_s5 += (c2 * (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dudrp2_s28 * a_stry[(j + 2)-jfirst+1] * istrx_s3 +
        a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 -
        (a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dudrm2_s25 * a_stry[(j - 2)-jfirst+1] * istrx_s3 +
         a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dvdrm2_s29)) +
      c1 * (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dudrp1_s27 * a_stry[(j + 1)-jfirst+1] * istrx_s3 +
          a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 -
          (a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dudrm1_s26 * a_stry[(j - 1)-jfirst+1] * istrx_s3 +
           a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dvdrm1_s30)))
r2_s6 += (c2 * (a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(2)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dudrp2_s28 +
        (2 * a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1]) *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 *
        a_stry[(j + 2)-jfirst+1] * istrx_s3 +
        a_lambda[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(4)+1] *
        a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 * istrx_s3 -
        (a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(2)+1] *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dudrm2_s25 +
         (2 * a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1]) *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dvdrm2_s29 *
         a_stry[(j - 2)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(4)+1] *
         a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dwdrm2_s33 * istrx_s3)) +
      c1 * (a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(2)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dudrp1_s27 +
          (2 * a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1]) *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 *
          a_stry[(j + 1)-jfirst+1] * istrx_s3 +
          a_lambda[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 * istrx_s3 -
          (a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(2)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dudrm1_s26 +
           (2 * a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] + a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1]) *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dvdrm1_s30 *
           a_stry[(j - 1)-jfirst+1] * istrx_s3 +
           a_lambda[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dwdrm1_s34 * istrx_s3)))
r3_s7 += ((c2 * (a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dwdrp2_s36 * a_stry[(j + 2)-jfirst+1] +
          a_mu[base+(i)+ni*( j + 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j + 2)+nij*( k)+nijk*(1)+1] * dvdrp2_s32 -
          (a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dwdrm2_s33 * a_stry[(j - 2)-jfirst+1] +
           a_mu[base+(i)+ni*( j - 2)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j - 2)+nij*( k)+nijk*(1)+1] * dvdrm2_s29)) +
        c1 * (a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(3)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dwdrp1_s35 * a_stry[(j + 1)-jfirst+1] +
          a_mu[base+(i)+ni*( j + 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(4)+1] *
          a_met[base-nijk+( i)+ni*( j + 1)+nij*( k)+nijk*(1)+1] * dvdrp1_s31 -
          (a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(3)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dwdrm1_s34 * a_stry[(j - 1)-jfirst+1] +
           a_mu[base+(i)+ni*( j - 1)+nij*( k)+1] * a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(4)+1] *
           a_met[base-nijk+( i)+ni*( j - 1)+nij*( k)+nijk*(1)+1] * dvdrm1_s30))) *
      istrx_s3)
for q_s39 in Int32(nk - 7):Int32(nk)
r1_s5 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s39 + 1)-1)+1] *
        (
         
         (2 * a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] + a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2 +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1])) +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1])) *
         istry_s2
         
         + a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(1)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(2)+1]))))
r2_s6 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s39 + 1)-1)+1] *
        (
         
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1])) +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(2)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2
         
         + a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(1)+1])) +
         (2 * a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] + a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1]) * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] *
         a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(2)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(3)+1])) *
         istrx_s3))
r3_s7 -= (a_bope[(nk - k + 1)-1+6*(( nk - q_s39 + 1)-1)+1] *
        (
         
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(1)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(1)+1])) *
         istry_s2 +
         a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(2)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i + 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 2)+ni*( j)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i + 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i - 1)+ni*( j)+nij*( q_s39)+nijk*(3)+1])) *
         a_strx[(i)-ifirst+1] * istry_s2
         
         + a_mu[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(3)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(3)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(3)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(3)+1])) *
         a_stry[(j)-jfirst+1] * istrx_s3 +
         a_lambda[base+(i)+ni*( j)+nij*( q_s39)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(4)+1] * a_met[base-nijk+( i)+ni*( j)+nij*( q_s39)+nijk*(1)+1] *
         (c2 * (a_u[base-nijk+( i)+ni*( j + 2)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 2)+nij*( q_s39)+nijk*(2)+1]) +
          c1 * (a_u[base-nijk+( i)+ni*( j + 1)+nij*( q_s39)+nijk*(2)+1] - a_u[base-nijk+( i)+ni*( j - 1)+nij*( q_s39)+nijk*(2)+1])) *
         istrx_s3))
end
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(1)+1] + sgn * r1_s5 * ijac_s1
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(2)+1] + sgn * r2_s6 * ijac_s1
a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] = a1 * a_lu[base-nijk+( i)+ni*( j)+nij*( k)+nijk*(3)+1] + sgn * r3_s7 * ijac_s1
end
return
end
