# IRIS batched GELS

Retains the source's two fixed square, nonsingular 5x5 systems, input arrays,
four real/complex precision modes, original tolerances and iteration/warmup CLI.
IRIS DMem HIP tasks solve each system by Gaussian elimination with partial
pivoting. Vendor GELS uses QR instead; this implementation has different
performance and roundoff and does not promise general rectangular GELS.
Every solution component is checked against the original exact all-ones answer.
The unchanged native HIP batched solver provides the independent comparison.
