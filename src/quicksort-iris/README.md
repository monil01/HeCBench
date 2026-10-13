IRIS dmem port of the CUDA two-stage GPU quicksort. Global partitions,
host work queues, local shared-memory quicksort/bitonic kernels, record layouts,
three element types, mt19937(19937) shuffle, and default run arguments are retained.
All partially written arrays and work records use read/write access. Each sort
creates fresh dmem objects after the original input is reset.

The original full element-by-element TRUST_BUT_VERIFY checks remain enabled;
IRIS returns failure when those checks fail. Native siblings remain unchanged.
Validation uses their original checks on bounded arguments 2 64 64 and 1 256 256.
CUDA kernel wrappers are provided but are not validated on this AMD machine.

