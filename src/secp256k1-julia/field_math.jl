# Direct scalar transcription of original secp256k1 10x26 field arithmetic.
# UInt32/UInt64 fixed-width wrap and unsigned casts preserve native semantics.

function fe_sqr(a::NTuple{10,UInt32})
 M = (UInt64(0x3FFFFFF)) % UInt32
 R0 = (UInt64(0x3D10)) % UInt32
 R1 = (UInt64(0x400)) % UInt32
 d = (UInt64((a[1]*UInt32(2))) * a[10]
    + UInt64((a[2]*UInt32(2))) * a[9]
    + UInt64((a[3]*UInt32(2))) * a[8]
    + UInt64((a[4]*UInt32(2))) * a[7]
    + UInt64((a[5]*UInt32(2))) * a[6]) % UInt64
 t9 = (d & M) % UInt32
 d = (d >> (UInt32(26))) % UInt64
 c = (UInt64(a[1]) * a[1]) % UInt64
 d = (d + (UInt64((a[2]*UInt32(2))) * a[10]
    + UInt64((a[3]*UInt32(2))) * a[9]
    + UInt64((a[4]*UInt32(2))) * a[8]
    + UInt64((a[5]*UInt32(2))) * a[7]
    + UInt64(a[6]) * a[6])) % UInt64
 u0 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u0 * R0)) % UInt64
 t0 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u0 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[2])) % UInt64
 d = (d + (UInt64((a[3]*UInt32(2))) * a[10]
    + UInt64((a[4]*UInt32(2))) * a[9]
    + UInt64((a[5]*UInt32(2))) * a[8]
    + UInt64((a[6]*UInt32(2))) * a[7])) % UInt64
 u1 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u1 * R0)) % UInt64
 t1 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u1 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[3]
    + UInt64(a[2]) * a[2])) % UInt64
 d = (d + (UInt64((a[4]*UInt32(2))) * a[10]
    + UInt64((a[5]*UInt32(2))) * a[9]
    + UInt64((a[6]*UInt32(2))) * a[8]
    + UInt64(a[7]) * a[7])) % UInt64
 u2 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u2 * R0)) % UInt64
 t2 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u2 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[4]
    + UInt64((a[2]*UInt32(2))) * a[3])) % UInt64
 d = (d + (UInt64((a[5]*UInt32(2))) * a[10]
    + UInt64((a[6]*UInt32(2))) * a[9]
    + UInt64((a[7]*UInt32(2))) * a[8])) % UInt64
 u3 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u3 * R0)) % UInt64
 t3 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u3 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[5]
    + UInt64((a[2]*UInt32(2))) * a[4]
    + UInt64(a[3]) * a[3])) % UInt64
 d = (d + (UInt64((a[6]*UInt32(2))) * a[10]
    + UInt64((a[7]*UInt32(2))) * a[9]
    + UInt64(a[8]) * a[8])) % UInt64
 u4 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u4 * R0)) % UInt64
 t4 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u4 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[6]
    + UInt64((a[2]*UInt32(2))) * a[5]
    + UInt64((a[3]*UInt32(2))) * a[4])) % UInt64
 d = (d + (UInt64((a[7]*UInt32(2))) * a[10]
    + UInt64((a[8]*UInt32(2))) * a[9])) % UInt64
 u5 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u5 * R0)) % UInt64
 t5 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u5 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[7]
    + UInt64((a[2]*UInt32(2))) * a[6]
    + UInt64((a[3]*UInt32(2))) * a[5]
    + UInt64(a[4]) * a[4])) % UInt64
 d = (d + (UInt64((a[8]*UInt32(2))) * a[10]
    + UInt64(a[9]) * a[9])) % UInt64
 u6 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u6 * R0)) % UInt64
 t6 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u6 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[8]
    + UInt64((a[2]*UInt32(2))) * a[7]
    + UInt64((a[3]*UInt32(2))) * a[6]
    + UInt64((a[4]*UInt32(2))) * a[5])) % UInt64
 d = (d + (UInt64((a[9]*UInt32(2))) * a[10])) % UInt64
 u7 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u7 * R0)) % UInt64
 t7 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u7 * R1)) % UInt64
 c = (c + (UInt64((a[1]*UInt32(2))) * a[9]
    + UInt64((a[2]*UInt32(2))) * a[8]
    + UInt64((a[3]*UInt32(2))) * a[7]
    + UInt64((a[4]*UInt32(2))) * a[6]
    + UInt64(a[5]) * a[5])) % UInt64
 d = (d + (UInt64(a[10]) * a[10])) % UInt64
 u8 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u8 * R0)) % UInt64
 r3 = (t3) % UInt32
 r4 = (t4) % UInt32
 r5 = (t5) % UInt32
 r6 = (t6) % UInt32
 r7 = (t7) % UInt32
 r8 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u8 * R1)) % UInt64
 c = (c + (d * R0 + t9)) % UInt64
 r9 = (c & (M >> UInt32(4))) % UInt32
 c = (c >> (UInt32(22))) % UInt64
 c = (c + (d * (R1 << UInt32(4)))) % UInt64
 d = (c * (R0 >> UInt32(4)) + t0) % UInt64
 r0 = (d & M) % UInt32
 d = (d >> (UInt32(26))) % UInt64
 d = (d + (c * (R1 >> UInt32(4)) + t1)) % UInt64
 r1 = (d & M) % UInt32
 d = (d >> (UInt32(26))) % UInt64
 d = (d + (t2)) % UInt64
 r2 = (d) % UInt32
 (r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,)
end

function fe_mul(a::NTuple{10,UInt32},b::NTuple{10,UInt32})
 M = (UInt64(0x3FFFFFF)) % UInt32
 R0 = (UInt64(0x3D10)) % UInt32
 R1 = (UInt64(0x400)) % UInt32
 d = (UInt64(a[1]) * b[10]
    + UInt64(a[2]) * b[9]
    + UInt64(a[3]) * b[8]
    + UInt64(a[4]) * b[7]
    + UInt64(a[5]) * b[6]
    + UInt64(a[6]) * b[5]
    + UInt64(a[7]) * b[4]
    + UInt64(a[8]) * b[3]
    + UInt64(a[9]) * b[2]
    + UInt64(a[10]) * b[1]) % UInt64
 t9 = (d & M) % UInt32
 d = (d >> (UInt32(26))) % UInt64
 c = (UInt64(a[1]) * b[1]) % UInt64
 d = (d + (UInt64(a[2]) * b[10]
    + UInt64(a[3]) * b[9]
    + UInt64(a[4]) * b[8]
    + UInt64(a[5]) * b[7]
    + UInt64(a[6]) * b[6]
    + UInt64(a[7]) * b[5]
    + UInt64(a[8]) * b[4]
    + UInt64(a[9]) * b[3]
    + UInt64(a[10]) * b[2])) % UInt64
 u0 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u0 * R0)) % UInt64
 t0 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u0 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[2]
    + UInt64(a[2]) * b[1])) % UInt64
 d = (d + (UInt64(a[3]) * b[10]
    + UInt64(a[4]) * b[9]
    + UInt64(a[5]) * b[8]
    + UInt64(a[6]) * b[7]
    + UInt64(a[7]) * b[6]
    + UInt64(a[8]) * b[5]
    + UInt64(a[9]) * b[4]
    + UInt64(a[10]) * b[3])) % UInt64
 u1 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u1 * R0)) % UInt64
 t1 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u1 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[3]
    + UInt64(a[2]) * b[2]
    + UInt64(a[3]) * b[1])) % UInt64
 d = (d + (UInt64(a[4]) * b[10]
    + UInt64(a[5]) * b[9]
    + UInt64(a[6]) * b[8]
    + UInt64(a[7]) * b[7]
    + UInt64(a[8]) * b[6]
    + UInt64(a[9]) * b[5]
    + UInt64(a[10]) * b[4])) % UInt64
 u2 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u2 * R0)) % UInt64
 t2 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u2 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[4]
    + UInt64(a[2]) * b[3]
    + UInt64(a[3]) * b[2]
    + UInt64(a[4]) * b[1])) % UInt64
 d = (d + (UInt64(a[5]) * b[10]
    + UInt64(a[6]) * b[9]
    + UInt64(a[7]) * b[8]
    + UInt64(a[8]) * b[7]
    + UInt64(a[9]) * b[6]
    + UInt64(a[10]) * b[5])) % UInt64
 u3 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u3 * R0)) % UInt64
 t3 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u3 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[5]
    + UInt64(a[2]) * b[4]
    + UInt64(a[3]) * b[3]
    + UInt64(a[4]) * b[2]
    + UInt64(a[5]) * b[1])) % UInt64
 d = (d + (UInt64(a[6]) * b[10]
    + UInt64(a[7]) * b[9]
    + UInt64(a[8]) * b[8]
    + UInt64(a[9]) * b[7]
    + UInt64(a[10]) * b[6])) % UInt64
 u4 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u4 * R0)) % UInt64
 t4 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u4 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[6]
    + UInt64(a[2]) * b[5]
    + UInt64(a[3]) * b[4]
    + UInt64(a[4]) * b[3]
    + UInt64(a[5]) * b[2]
    + UInt64(a[6]) * b[1])) % UInt64
 d = (d + (UInt64(a[7]) * b[10]
    + UInt64(a[8]) * b[9]
    + UInt64(a[9]) * b[8]
    + UInt64(a[10]) * b[7])) % UInt64
 u5 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u5 * R0)) % UInt64
 t5 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u5 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[7]
    + UInt64(a[2]) * b[6]
    + UInt64(a[3]) * b[5]
    + UInt64(a[4]) * b[4]
    + UInt64(a[5]) * b[3]
    + UInt64(a[6]) * b[2]
    + UInt64(a[7]) * b[1])) % UInt64
 d = (d + (UInt64(a[8]) * b[10]
    + UInt64(a[9]) * b[9]
    + UInt64(a[10]) * b[8])) % UInt64
 u6 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u6 * R0)) % UInt64
 t6 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u6 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[8]
    + UInt64(a[2]) * b[7]
    + UInt64(a[3]) * b[6]
    + UInt64(a[4]) * b[5]
    + UInt64(a[5]) * b[4]
    + UInt64(a[6]) * b[3]
    + UInt64(a[7]) * b[2]
    + UInt64(a[8]) * b[1])) % UInt64
 d = (d + (UInt64(a[9]) * b[10]
    + UInt64(a[10]) * b[9])) % UInt64
 u7 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u7 * R0)) % UInt64
 t7 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u7 * R1)) % UInt64
 c = (c + (UInt64(a[1]) * b[9]
    + UInt64(a[2]) * b[8]
    + UInt64(a[3]) * b[7]
    + UInt64(a[4]) * b[6]
    + UInt64(a[5]) * b[5]
    + UInt64(a[6]) * b[4]
    + UInt64(a[7]) * b[3]
    + UInt64(a[8]) * b[2]
    + UInt64(a[9]) * b[1])) % UInt64
 d = (d + (UInt64(a[10]) * b[10])) % UInt64
 u8 = (d & M) % UInt64
 d = (d >> (UInt32(26))) % UInt64
 c = (c + (u8 * R0)) % UInt64
 r3 = (t3) % UInt32
 r4 = (t4) % UInt32
 r5 = (t5) % UInt32
 r6 = (t6) % UInt32
 r7 = (t7) % UInt32
 r8 = (c & M) % UInt32
 c = (c >> (UInt32(26))) % UInt64
 c = (c + (u8 * R1)) % UInt64
 c = (c + (d * R0 + t9)) % UInt64
 r9 = (c & (M >> UInt32(4))) % UInt32
 c = (c >> (UInt32(22))) % UInt64
 c = (c + (d * (R1 << UInt32(4)))) % UInt64
 d = (c * (R0 >> UInt32(4)) + t0) % UInt64
 r0 = (d & M) % UInt32
 d = (d >> (UInt32(26))) % UInt64
 d = (d + (c * (R1 >> UInt32(4)) + t1)) % UInt64
 r1 = (d & M) % UInt32
 d = (d >> (UInt32(26))) % UInt64
 d = (d + (t2)) % UInt64
 r2 = (d) % UInt32
 (r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,)
end

function fe_bytes(a::NTuple{10,UInt32})
 r0 = ((a[10] >> UInt32(14)) & UInt32(0xff)) % UInt8
 r1 = ((a[10] >> UInt32(6)) & UInt32(0xff)) % UInt8
 r2 = (((a[10] & UInt32(0x3F)) << UInt32(2)) | ((a[9] >> UInt32(24)) & UInt32(0x3))) % UInt8
 r3 = ((a[9] >> UInt32(16)) & UInt32(0xff)) % UInt8
 r4 = ((a[9] >> UInt32(8)) & UInt32(0xff)) % UInt8
 r5 = (a[9] & UInt32(0xff)) % UInt8
 r6 = ((a[8] >> UInt32(18)) & UInt32(0xff)) % UInt8
 r7 = ((a[8] >> UInt32(10)) & UInt32(0xff)) % UInt8
 r8 = ((a[8] >> UInt32(2)) & UInt32(0xff)) % UInt8
 r9 = (((a[8] & UInt32(0x3)) << UInt32(6)) | ((a[7] >> UInt32(20)) & UInt32(0x3f))) % UInt8
 r10 = ((a[7] >> UInt32(12)) & UInt32(0xff)) % UInt8
 r11 = ((a[7] >> UInt32(4)) & UInt32(0xff)) % UInt8
 r12 = (((a[7] & UInt32(0xf)) << UInt32(4)) | ((a[6] >> UInt32(22)) & UInt32(0xf))) % UInt8
 r13 = ((a[6] >> UInt32(14)) & UInt32(0xff)) % UInt8
 r14 = ((a[6] >> UInt32(6)) & UInt32(0xff)) % UInt8
 r15 = (((a[6] & UInt32(0x3f)) << UInt32(2)) | ((a[5] >> UInt32(24)) & UInt32(0x3))) % UInt8
 r16 = ((a[5] >> UInt32(16)) & UInt32(0xff)) % UInt8
 r17 = ((a[5] >> UInt32(8)) & UInt32(0xff)) % UInt8
 r18 = (a[5] & UInt32(0xff)) % UInt8
 r19 = ((a[4] >> UInt32(18)) & UInt32(0xff)) % UInt8
 r20 = ((a[4] >> UInt32(10)) & UInt32(0xff)) % UInt8
 r21 = ((a[4] >> UInt32(2)) & UInt32(0xff)) % UInt8
 r22 = (((a[4] & UInt32(0x3)) << UInt32(6)) | ((a[3] >> UInt32(20)) & UInt32(0x3f))) % UInt8
 r23 = ((a[3] >> UInt32(12)) & UInt32(0xff)) % UInt8
 r24 = ((a[3] >> UInt32(4)) & UInt32(0xff)) % UInt8
 r25 = (((a[3] & UInt32(0xf)) << UInt32(4)) | ((a[2] >> UInt32(22)) & UInt32(0xf))) % UInt8
 r26 = ((a[2] >> UInt32(14)) & UInt32(0xff)) % UInt8
 r27 = ((a[2] >> UInt32(6)) & UInt32(0xff)) % UInt8
 r28 = (((a[2] & UInt32(0x3f)) << UInt32(2)) | ((a[1] >> UInt32(24)) & UInt32(0x3))) % UInt8
 r29 = ((a[1] >> UInt32(16)) & UInt32(0xff)) % UInt8
 r30 = ((a[1] >> UInt32(8)) & UInt32(0xff)) % UInt8
 r31 = (a[1] & UInt32(0xff)) % UInt8
 (r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,)
end

@inline function fe_storage(a::NTuple{8,UInt32})
 m=UInt32(0x3ffffff)
 (a[1]&m,(a[1]>>26)|((a[2]<<6)&m),(a[2]>>20)|((a[3]<<12)&m),(a[3]>>14)|((a[4]<<18)&m),(a[4]>>8)|((a[5]<<24)&m),(a[5]>>2)&m,(a[5]>>28)|((a[6]<<4)&m),(a[6]>>22)|((a[7]<<10)&m),(a[7]>>16)|((a[8]<<16)&m),a[8]>>10)
end
@inline function fe_weak(a::NTuple{10,UInt32})
 m=UInt32(0x3ffffff);x=a[10]>>22
 t0=a[1]+x*UInt32(0x3d1);t1=a[2]+(x<<6)+(t0>>26);t2=a[3]+(t1>>26);t3=a[4]+(t2>>26);t4=a[5]+(t3>>26);t5=a[6]+(t4>>26);t6=a[7]+(t5>>26);t7=a[8]+(t6>>26);t8=a[9]+(t7>>26);t9=(a[10]&UInt32(0x3fffff))+(t8>>26)
 (t0&m,t1&m,t2&m,t3&m,t4&m,t5&m,t6&m,t7&m,t8&m,t9)
end
@inline fe_add(a::NTuple{10,UInt32},b::NTuple{10,UInt32})=ntuple(i->a[i]+b[i],Val(10))
@inline fe_int(a::NTuple{10,UInt32},n::UInt32)=ntuple(i->a[i]*n,Val(10))
@inline function fe_neg(a::NTuple{10,UInt32},m::UInt32)
 ntuple(i->(i==1 ? UInt32(0x3fffc2f) : i==2 ? UInt32(0x3ffffbf) : i==10 ? UInt32(0x3fffff) : UInt32(0x3ffffff))*UInt32(2)*(m+UInt32(1))-a[i],Val(10))
end
@inline fe_one()=(UInt32(1),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0))
function fe_squares(a,n)
 for _ in 1:n;a=fe_sqr(a);end
 a
end
function fe_inv(a)
 x2=fe_mul(fe_sqr(a),a);x3=fe_mul(fe_sqr(x2),a)
 x6=fe_mul(fe_squares(x3,3),x3);x9=fe_mul(fe_squares(x6,3),x3);x11=fe_mul(fe_squares(x9,2),x2)
 x22=fe_mul(fe_squares(x11,11),x11);x44=fe_mul(fe_squares(x22,22),x22);x88=fe_mul(fe_squares(x44,44),x44)
 x176=fe_mul(fe_squares(x88,88),x88);x220=fe_mul(fe_squares(x176,44),x44);x223=fe_mul(fe_squares(x220,3),x3)
 t=fe_mul(fe_squares(x223,23),x22);t=fe_mul(fe_squares(t,5),a);t=fe_mul(fe_squares(t,3),x2);fe_mul(a,fe_squares(t,2))
end
function ge_add(x,y,z,bx,by)
 z12=fe_sqr(z);u1=fe_weak(x);u2=fe_mul(bx,z12);s1=fe_weak(y);s2=fe_mul(fe_mul(by,z12),z)
 h=fe_add(fe_neg(u1,UInt32(1)),u2);ii=fe_add(fe_neg(s1,UInt32(1)),s2)
 i2=fe_sqr(ii);h2=fe_sqr(h);h3=fe_mul(h,h2);znew=fe_mul(z,h);t=fe_mul(u1,h2)
 xx=fe_add(fe_neg(fe_add(fe_int(t,UInt32(2)),h3),UInt32(3)),i2)
 yy=fe_mul(fe_add(fe_neg(xx,UInt32(5)),t),ii);yy=fe_add(yy,fe_neg(fe_mul(h3,s1),UInt32(1)))
 xx,yy,znew
end
@inline function load_point(input,offset)
 @inbounds x=fe_storage(ntuple(j->input[offset+j],Val(8)))
 @inbounds y=fe_storage(ntuple(j->input[offset+8+j],Val(8)))
 x,y
end
function secp_compute(input)
 x,y=load_point(input,Int32(0));z=fe_one();zall=z
 for point in Int32(1):Int32(511)
  bx,by=load_point(input,point*Int32(16));x,y,z=ge_add(x,y,z,bx,by);zall=fe_mul(zall,z)
 end
 fe_bytes(fe_inv(zall))
end
