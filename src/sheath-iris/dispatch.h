#pragma once
inline void dispatch_scatterParticle(dim3 grid, dim3 block, const Particle* __restrict particles, float*__restrict den, long N) { iris_launch("scatterParticle", grid, block, particles, den, N); }
inline void dispatch_pushParticle(dim3 grid, dim3 block, Particle*__restrict particles, const double*__restrict ef, double qm, long N) { iris_launch("pushParticle", grid, block, particles, ef, qm, N); }
inline void dispatch_rewindParticle(dim3 grid, dim3 block, Particle*__restrict particles, const double*__restrict ef, double qm, long N) { iris_launch("rewindParticle", grid, block, particles, ef, qm, N); }
