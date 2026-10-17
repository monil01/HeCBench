#=
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 =#


# Native butterfly arithmetic translated once to immutable Julia tuples.
const KD_C_a = Float32(1.3870398453221474618216191915664)
const KD_C_b = Float32(1.3065629648763765278566431734272)
const KD_C_c = Float32(1.1758756024193587169744671046113)
const KD_C_d = Float32(0.78569495838710218127789736765722)
const KD_C_e = Float32(0.54119610014619698439972320536639)
const KD_C_f = Float32(0.27589937928294301233595756366937)
const KD_C_norm = Float32(0.35355339059327376220042218105242)
@inline function study_dct8(d)
 X07P = d[1] + d[8]
 X16P = d[2] + d[7]
 X25P = d[3] + d[6]
 X34P = d[4] + d[5]
 X07M = d[1] - d[8]
 X61M = d[7] - d[2]
 X25M = d[3] - d[6]
 X43M = d[5] - d[4]
 X07P34PP = X07P + X34P
 X07P34PM = X07P - X34P
 X16P25PP = X16P + X25P
 X16P25PM = X16P - X25P
 z0 = KD_C_norm * (X07P34PP + X16P25PP)
 z2 = KD_C_norm * (KD_C_b * X07P34PM + KD_C_e * X16P25PM)
 z4 = KD_C_norm * (X07P34PP - X16P25PP)
 z6 = KD_C_norm * (KD_C_e * X07P34PM - KD_C_b * X16P25PM)
 z1 = KD_C_norm * (KD_C_a * X07M - KD_C_c * X61M + KD_C_d * X25M - KD_C_f * X43M)
 z3 = KD_C_norm * (KD_C_c * X07M + KD_C_f * X61M - KD_C_a * X25M + KD_C_d * X43M)
 z5 = KD_C_norm * (KD_C_d * X07M + KD_C_a * X61M + KD_C_f * X25M - KD_C_c * X43M)
 z7 = KD_C_norm * (KD_C_f * X07M + KD_C_d * X61M + KD_C_c * X25M + KD_C_a * X43M)
 (z0,z1,z2,z3,z4,z5,z6,z7)
end
@inline function study_idct8(d)
 Y04P = d[1] + d[5]
 Y2b6eP = KD_C_b * d[3] + KD_C_e * d[7]
 Y04P2b6ePP = Y04P + Y2b6eP
 Y04P2b6ePM = Y04P - Y2b6eP
 Y7f1aP3c5dPP = KD_C_f * d[8] + KD_C_a * d[2] + KD_C_c * d[4] + KD_C_d * d[6]
 Y7a1fM3d5cMP = KD_C_a * d[8] - KD_C_f * d[2] + KD_C_d * d[4] - KD_C_c * d[6]
 Y04M = d[1] - d[5]
 Y2e6bM = KD_C_e * d[3] - KD_C_b * d[7]
 Y04M2e6bMP = Y04M + Y2e6bM
 Y04M2e6bMM = Y04M - Y2e6bM
 Y1c7dM3f5aPM = KD_C_c * d[2] - KD_C_d * d[8] - KD_C_f * d[4] - KD_C_a * d[6]
 Y1d7cP3a5fMM = KD_C_d * d[2] + KD_C_c * d[8] - KD_C_a * d[4] + KD_C_f * d[6]
 z0 = KD_C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP)
 z7 = KD_C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP)
 z4 = KD_C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP)
 z3 = KD_C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP)
 z1 = KD_C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM)
 z5 = KD_C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM)
 z2 = KD_C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM)
 z6 = KD_C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM)
 (z0,z1,z2,z3,z4,z5,z6,z7)
end
@inline study_butterfly(d,dir)=dir==666 ? study_dct8(d) : study_idct8(d)
