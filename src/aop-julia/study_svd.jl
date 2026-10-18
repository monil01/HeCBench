# Arithmetic expression translation of original NVIDIA QR/Jacobi routines; default compressed W. See reference_original/native_main.cu.
@inline function assemble_R(m,sums,smem_svds)
x0 = smem_svds[1]
x1 = smem_svds[2]
x2 = smem_svds[3]
x0_sq = x0 * x0
sum1 = sums[1] - x0
sum2 = sums[2] - x0_sq
sum3 = sums[3] - x0_sq*x0
sum4 = sums[4] - x0_sq*x0_sq
m_as_dbl = Float64(m)
sigma = m_as_dbl - 1.0
mu = sqrt(m_as_dbl)
v0 = -sigma / (1.0 + mu)
v0_sq = v0*v0
beta = 2.0 * v0_sq / (sigma + v0_sq)
inv_v0 = 1.0 / v0
one_min_beta = 1.0 - beta
beta_div_v0  = beta * inv_v0
smem_svds[1] = mu
smem_svds[2] = one_min_beta*x0 - beta_div_v0*sum1
smem_svds[3] = one_min_beta*x0_sq - beta_div_v0*sum2
beta_div_v0_sq = beta_div_v0 * inv_v0
c1 = beta_div_v0_sq*sum1 + beta_div_v0*x0
c2 = beta_div_v0_sq*sum2 + beta_div_v0*x0_sq
x1_sq = x1*x1
sum1 -= x1
sum2 -= x1_sq
sum3 -= x1_sq*x1
sum4 -= x1_sq*x1_sq
x0 = x1-c1
x0_sq = x0*x0
sigma = sum2 - 2.0*c1*sum1 + (m_as_dbl-2.0)*c1*c1
if abs(sigma) < 1.0e-16
beta = 0.0
else
mu = sqrt(x0_sq + sigma)
if x0 <= 0.0
v0 = x0 - mu
else
v0 = -sigma / (x0 + mu)
end
v0_sq = v0*v0
beta = 2.0*v0_sq / (sigma + v0_sq)
end
inv_v0 = 1.0 / v0
beta_div_v0 = beta * inv_v0
c3 = (sum3 - c1*sum2 - c2*sum1 + (m_as_dbl-2.0)*c1*c2)*beta_div_v0
c4 = (x1_sq-c2)*beta_div_v0 + c3*inv_v0
c5 = c1*c4 - c2
one_min_beta = 1.0 - beta
smem_svds[4] = one_min_beta*x0 - beta_div_v0*sigma
smem_svds[5] = one_min_beta*(x1_sq-c2) - c3
x2_sq = x2*x2
sum1 -= x2
sum2 -= x2_sq
sum3 -= x2_sq*x2
sum4 -= x2_sq*x2_sq
x0 = x2_sq-c4*x2+c5
sigma = sum4 - 2.0*c4*sum3 + (c4*c4 + 2.0*c5)*sum2 - 2.0*c4*c5*sum1 + (m_as_dbl-3.0)*c5*c5
if abs(sigma) < 1.0e-12
beta = 0.0
else
mu = sqrt(x0*x0 + sigma)
if x0 <= 0.0
v0 = x0 - mu
else
v0 = -sigma / (x0 + mu)
end
v0_sq = v0*v0
beta = 2.0*v0_sq / (sigma + v0_sq)
end
smem_svds[6] = (1.0-beta)*x0 - (beta/v0)*sigma
end
@inline function off_diag_norm(A01,A02,A12)
return sqrt(2.0 * (A01*A01 + A02*A02 + A12*A12))
end
@inline function svd_3x3(m,sums,smem_svds)
assemble_R(m, sums, smem_svds)
R00 = smem_svds[1]
R01 = smem_svds[2]
R02 = smem_svds[3]
R11 = smem_svds[4]
R12 = smem_svds[5]
R22 = smem_svds[6]
A00 = R00*R00
A01 = R00*R01
A02 = R00*R02
A11 = R01*R01 + R11*R11
A12 = R01*R02 + R11*R12
A22 = R02*R02 + R12*R12 + R22*R22
V00,V01,V02=1.0,0.0,0.0
V10,V11,V12=0.0,1.0,0.0
V20,V21,V22=0.0,0.0,1.0
max_iters = 16
tolerance = 1.0e-12
iter=0
while off_diag_norm(A01,A02,A12)>=tolerance && iter<max_iters
iter+=1
c,s=1.0,0.0
if A01 != 0.0
tau = (A11 - A00) / (2.0 * A01)
sgn = (tau < 0.0) ? -1.0 : 1.0
t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau))
c = 1.0 / sqrt(1.0 + t*t)
s = t*c
end
B00 = c*A00 - s*A01
B01 = s*A00 + c*A01
B10 = c*A01 - s*A11
B11 = s*A01 + c*A11
B02 = A02
A00 = c*B00 - s*B10
A01 = c*B01 - s*B11
A11 = s*B01 + c*B11
A02 = c*B02 - s*A12
A12 = s*B02 + c*A12
B00 = c*V00 - s*V01
V01 = s*V00 + c*V01
V00 = B00
B10 = c*V10 - s*V11
V11 = s*V10 + c*V11
V10 = B10
B20 = c*V20 - s*V21
V21 = s*V20 + c*V21
V20 = B20
c,s=1.0,0.0
if A02 != 0.0
tau = (A22 - A00) / (2.0 * A02)
sgn = (tau < 0.0) ? -1.0 : 1.0
t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau))
c = 1.0 / sqrt(1.0 + t*t)
s = t*c
end
B00 = c*A00 - s*A02
B01 = c*A01 - s*A12
B02 = s*A00 + c*A02
B20 = c*A02 - s*A22
B22 = s*A02 + c*A22
A00 = c*B00 - s*B20
A12 = s*A01 + c*A12
A02 = c*B02 - s*B22
A22 = s*B02 + c*B22
A01 = B01
B00 = c*V00 - s*V02
V02 = s*V00 + c*V02
V00 = B00
B10 = c*V10 - s*V12
V12 = s*V10 + c*V12
V10 = B10
B20 = c*V20 - s*V22
V22 = s*V20 + c*V22
V20 = B20
c,s=1.0,0.0
if A12 != 0.0
tau = (A22 - A11) / (2.0 * A12)
sgn = (tau < 0.0) ? -1.0 : 1.0
t   = sgn / (sgn*tau + sqrt(1.0 + tau*tau))
c = 1.0 / sqrt(1.0 + t*t)
s = t*c
end
B02 = s*A01 + c*A02
B11 = c*A11 - s*A12
B12 = s*A11 + c*A12
B21 = c*A12 - s*A22
B22 = s*A12 + c*A22
A01 = c*A01 - s*A02
A02 = B02
A11 = c*B11 - s*B21
A12 = c*B12 - s*B22
A22 = s*B12 + c*B22
B01 = c*V01 - s*V02
V02 = s*V01 + c*V02
V01 = B01
B11 = c*V11 - s*V12
V12 = s*V11 + c*V12
V11 = B11
B21 = c*V21 - s*V22
V22 = s*V21 + c*V22
V21 = B21
end
if A00 < A11
A00,A11=A11,A00
V00,V01=V01,V00
V10,V11=V11,V10
V20,V21=V21,V20
end
if A00 < A22
A00,A22=A22,A00
V00,V02=V02,V00
V10,V12=V12,V10
V20,V22=V22,V20
end
if A11 < A22
A11,A22=A22,A11
V01,V02=V02,V01
V11,V12=V12,V11
V21,V22=V22,V21
end
inv_S0 = (abs(A00) < 1.0e-12) ? 0.0 : 1.0 / A00
inv_S1 = (abs(A11) < 1.0e-12) ? 0.0 : 1.0 / A11
inv_S2 = (abs(A22) < 1.0e-12) ? 0.0 : 1.0 / A22
U00 = V00 * inv_S0
U01 = V01 * inv_S1
U02 = V02 * inv_S2
U10 = V10 * inv_S0
U11 = V11 * inv_S1
U12 = V12 * inv_S2
U20 = V20 * inv_S0
U21 = V21 * inv_S1
U22 = V22 * inv_S2
B00 = U00*V00 + U01*V01 + U02*V02
B01 = U00*V10 + U01*V11 + U02*V12
B02 = U00*V20 + U01*V21 + U02*V22
B11 = U10*V10 + U11*V11 + U12*V12
B12 = U10*V20 + U11*V21 + U12*V22
B22 = U20*V20 + U21*V21 + U22*V22
smem_svds[7] = B00*R00 + B01*R01 + B02*R02
smem_svds[8] =           B01*R11 + B02*R12
smem_svds[9] =                     B02*R22
smem_svds[10] =           B11*R11 + B12*R12
smem_svds[11] =                     B12*R22
smem_svds[12] =                     B22*R22
end
