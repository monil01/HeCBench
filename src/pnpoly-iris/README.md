Preserves the original 20,000,000 points, 600 vertices, random generator,
base kernel, seven tiled variants, and repeat argument. Every tiled result is
flushed and compared over the full bitmap against the base GPU result. Invalid
tail lanes are guarded in the IRIS kernel. Native HIP checksum and final PASS
are compared with `./main 1`; IRIS runs with IRIS_ARCHS=HIP.
