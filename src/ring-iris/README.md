# ring IRIS

This port uses one IRIS dmem object per detected IRIS HIP device and performs
the circular exchange with a copy kernel. The final copy back into buffer zero
is flushed to host memory for the same PASS/FAIL check as the native HIP
baseline.
