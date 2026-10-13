# Lookback option IRIS (loopback)
Preserves dump/repeat arguments, 4096x256 default grid,100 run repeats,
15 time steps and512 simulated paths per thread. Kiss/Rand initial states and
unseeded C rand initialization order retained before IRIS platform init. All
original1048576 parameter values and4194304 seed values generated even for
bounded validation. LOOPBACK_VALIDATION_BLOCKS optionally limits active blocks
while retaining original seed-plane stride and every per-thread simulation.
Only active outputs printed/dumped under bounded validation. Optional
LOOPBACK_VALIDATION_DUMP saves every active mean and variance for HIP comparison.
All memory is DMem; seed/parameters read-only, results written and flushed.
