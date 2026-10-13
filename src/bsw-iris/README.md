# Batched Smith-Waterman with IRIS DMem

Retains original protein loader/scoring/gaps,batched forward/reverse kernel and
existing native HIPwarp intrinsic lowering. Bool flag has1byte and gap scalars
2bytes; sequence offsets retain32bitunsigned. DMems read/write for partial
batch outputs; host prefix/string resets use update and fetch copies exactly
source's per-batch byte count. Dynamic validity scratch becomes3072staticbytes
for max1024threads, matching original kernel's3*minSize layout. Algorithms,
fullfileCLI,originalbatches and resultcolumns preserved.
Original DVC archive3.9MB recovered/MD5/size/entries verified in owned scratch.
Full30000-protein regression compared to native HIP and original result_aa;
bounded inputs are prefixes of original real FASTA. No native originals modified.
