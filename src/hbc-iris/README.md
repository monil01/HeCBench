IRIS DMem hybrid betweenness-centrality port preserving graph/verification/
approximation/score-output arguments and original graph algorithm/reference.
Default breast_coexpress.txt path and output.txt are retained, but the default
dataset is absent locally. Validation uses six-node chain and eight-node cycle.

Pitched scratch storage uses128-byte-aligned rows in1D DMem, with byte pitches
passed unchanged to the original indexing logic. Scratch pointers declared
const but written through kernel casts are explicitly rw. Graph inputs are
read-only, resets refresh host-backed DMem, scores flush before verification.
Native GPU queries only retain device0 geometry; all memory and compute use
IRIS. The unused empty source list gets a one-byte DMem placeholder.

Both graphs produce zero RMS/max CPU-reference error on native HIP and IRIS,
and all fourteen finite output scores match exactly. Large default graph and
approximation paths are preserved but not validated. Optional CUDA kernel
source is provided without CUDA validation. Original license is retained.
