# wordcount IRIS

This port preserves the CUDA/HIP word-start semantics without using Thrust.
The IRIS HIP kernel marks each character that starts a word into a dmem output
buffer, flushes that buffer to the host, and performs the final integer
reduction on the host.
