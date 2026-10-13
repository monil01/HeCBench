IRIS DMem lavaMD port preserving -boxes1d, default30boxes/direction, seed2,
100particles/box,128-thread geometry, alpha0.5, float force/charge arithmetic,
and original long box offsets/portable struct layout.

The real HIP force kernel uses typed dispatch. Input buffers are read-only,
initial zero forces explicitly update DMem and remain rw during accumulation,
and forces flush before optional OUTPUT=yes result.txt writing. No native
allocation/copy/kernel launch remains. Finalization precedes static teardown.

With OUTPUT=yes, both native HIP and IRIS validate2 and3boxes/direction:
all3200 and10800finite force scalars match exactly at printed precision
(rtol1e-5/atol1e-6). Default30is preserved but not GPU-validated in this pass.
An initial malformed C-linkage split was repaired using complete original
argument helpers; both attempts and full logs are retained in local state.
