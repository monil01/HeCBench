Logic-rewrite is deferred under the current restriction on opaque library
operation replacements. Although it owns many device kernels, the full resyn2
pipeline interleaves them with required host-dispatched Thrust GPU algorithms:
inclusive scans, copy_if/remove compaction, sequence, sort_by_key, counts,
reductions, and transform_reduce. For example, balance.cu, refactor.cu and
hash_table.h consume the resulting counts and reordered buffers immediately.

No IRIS dmem task binding for those host-dispatched operations is supplied.
Retaining only the device kernels would omit essential graph transformations.
Inventing equivalent scan/sort/compaction implementations is outside the
currently authorized scope; direct native Thrust launches would violate the
dmem-only host rule. Device-only Thrust helpers are not the blocker.

No executable IRIS port or numerical pass is claimed. The original hyp.aig
input, CLI, resyn2 command sequence, graph output semantics, and native HIP
sources are preserved. This deferral can be revisited with an approved task
binding or explicit authorization for library-operation replacements.
