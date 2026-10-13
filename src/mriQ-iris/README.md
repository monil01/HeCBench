# MRI-Q IRIS
Preserves two binary input/output filename arguments, file format, 256-thread
blocks, 1024-k tile accumulation and output layout. DMem owns all host arrays;
phi magnitudes flushed for host kValues assembly; Qr/Qi start at zero and use
iris_rw across tile tasks. Constant-memory tiles represented by a read-only
DMem kValues buffer (P1 performance compromise). Explicit pixel guard prevents
rounded-grid overflow. Odd tile count uses tile parity to avoid original odd-
numK cross-tile read past constant array; original well-defined even tiles and
single odd tile arithmetic order retained. Fast-math matches native HIP flags.
