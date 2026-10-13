# LCI IRIS DMem port

Retains original 64-mode equations, 0.1-to-200 time sweep, seeded mt19937 input,
and 96-thread geometry. Host reset is published through DMem.update; partially
written RHS uses read/write access. DUMP=yes retains original four-mode CSV.
Optional HECBENCH_RHS_OUTPUT writes every RHS mode 0..63 as binary doubles for
all timesteps (the unwritten truncation slot 64 is intentionally excluded).
Native HIP isolated adapter adds only this dump and uses the CUDA kernel/table
headers missing from the sibling verbatim. No native originals are modified.
At the default 64 modes, native high-order RHS outputs in modes 25..63 are
nonfinite. Verification requires identical NaN/Inf masks and compares every
finite value (49975 over the full sweep); equations and default modes are
preserved. This port reproduces native behavior, including that limitation.
