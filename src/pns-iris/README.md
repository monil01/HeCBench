IRIS DMem Petri-net port preserving N/S/T arguments, default22/100000/1000000,
750MB memory cap,256-thread trajectories, Mersenne Twister implementation and
5489*(batch+1) seeds, and four output statistics.

The original RNG and packed-place kernel compile as a real HIP code object.
All allocations/copies/launches use IRIS data memory/tasks. Packed state stays
rw, statistical buffers flush, and finalization precedes static destruction.

Validated N22/S624/T64 against native HIP: all four finite statistics match
exactly at printed precision. Large default trajectories are preserved but
not validated in this bounded run. No CPU backend fallback is used.
