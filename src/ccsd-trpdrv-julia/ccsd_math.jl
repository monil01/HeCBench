# Equation body copied from preserved genuine CUDA.jl ccsd_kernel!, original reference phase.
@inline function ccsd_terms(b,c,f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,eaijk,ncor,nocc,nvir)
        bc = b + (c - Int32(1)) * nvir
        cb = c + (b - Int32(1)) * nvir
        denom = -1.0 / (eorb[ncor + nocc + b] + eorb[ncor + nocc + c] + eaijk)

        f1nbc = f1n[bc]; f1tbc = f1t[bc]; f1ncb = f1n[cb]; f1tcb = f1t[cb]
        f2nbc = f2n[bc]; f2tbc = f2t[bc]; f2ncb = f2n[cb]; f2tcb = f2t[cb]
        f3nbc = f3n[bc]; f3tbc = f3t[bc]; f3ncb = f3n[cb]; f3tcb = f3t[cb]
        f4nbc = f4n[bc]; f4tbc = f4t[bc]; f4ncb = f4n[cb]; f4tcb = f4t[cb]

        s1 = denom * (f1tbc + f1ncb + f2tcb + f3nbc + f4ncb) * (f1tbc - f2tbc * 2 - f3tbc * 2 + f4tbc) -
             denom * (f1nbc + f1tcb + f2ncb + f3ncb) * (f1tbc * 2 - f2tbc - f3tbc + f4tbc * 2) +
             denom * 3 * (f1nbc * (f1nbc + f3ncb + f4tcb * 2) + f2nbc * f2tcb + f3nbc * f4tbc)

        s2 = denom * (f1nbc + f1tcb + f2ncb + f3tbc + f4tcb) * (f1nbc - f2nbc * 2 - f3nbc * 2 + f4nbc) -
             denom * (f1tbc + f1ncb + f2tcb + f3tcb) * (f1nbc * 2 - f2nbc - f3nbc + f4nbc * 2) +
             denom * 3 * (f1tbc * (f1tbc + f3tcb + f4ncb * 2) + f2tbc * f2ncb + f3tbc * f4nbc)

        t1v1b = t1v1[b]; t1v2b = t1v2[b]
        dintx1c = dintx1[c]; dintx2c = dintx2[c]; dintc1c = dintc1[c]; dintc2c = dintc2[c]

        s3 = denom * t1v1b * dintx1c * (f1tbc + f2nbc + f4ncb - (f3tbc + f4nbc + f2ncb + f1nbc + f2tbc + f3ncb) * 2 +
             (f3nbc + f4tbc + f1ncb) * 4) +
             denom * t1v1b * dintc1c * (f1nbc + f4nbc + f1tcb - (f2nbc + f3nbc + f2tcb) * 2)

        s4 = denom * t1v2b * dintx2c * (f1nbc + f2tbc + f4tcb - (f3nbc + f4tbc + f2tcb + f1tbc + f2nbc + f3tcb) * 2 +
             (f3tbc + f4nbc + f1tcb) * 4) +
             denom * t1v2b * dintc2c * (f1tbc + f4tbc + f1ncb - (f2tbc + f3tbc + f2ncb) * 2)


 return s1,s3,s2,s4
end
