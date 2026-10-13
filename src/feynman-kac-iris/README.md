# Feynman-Kac IRIS
Retains repeat argument/run10,1000 trajectories per point,nj128/ni255 grid,
seed123456789,a2/b1,h0.001,16x16 blocks, deterministic per-thread RNG and
analytic RMS calculation. DMem scalar accumulators use iris_rw and explicit
zero/reset plus DMem.update(host) each repeat; outputs flushed only at end.
FK_VALIDATION_NJ optionally reduces grid resolution while retaining1000 paths,
seed,stepsize and equations. FK_VALIDATION_SUMMARY reports count and raw error
sum for strong native comparison. Atomic double sum allows reduction-order
roundoff; validation uses rtol1e-10/atol1e-12 and exact inside count.
