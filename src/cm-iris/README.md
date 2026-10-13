IRIS dmem port retaining original connectivity-map query/profile parsing,
compound grouping, gene count22283, RNG engine seed123, random generations100000,
score normalization, p-value calculation, output names and directory argument.
Host ref/random changes call DMem.update. Kernel UCmax and threshold retain
native float widths even though host calculations use double. Equivalent static
128-int reduction storage matches the original fixed128-thread launch.

Default downloaded cudaMap dataset is absent. Validation supplies complete
deterministic22283-gene reference profiles and queries through the original
directory argument, with defaults unchanged. Native baseline sources are
unmodified; all result artifact rows are compared, including compound grouping
and p-values. HIP code-object path is resolved before source directory changes.
CUDA backend retained but unvalidated on NVIDIA.

