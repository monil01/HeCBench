# Distance-driven backprojection with IRIS DMem

Original CUDA defaults and geometry are retained. Host padding is published with
DMem.update; tasks retain original grid/block geometry, including the original
unsynchronized scan behavior. All buffers use DMem read/write access. Detector
subview uses an explicit int offset scalar in interpolation. HIP AMD GPU required.

Optional validation environment variables nPixX,nPixY,nSlices,nDetX,nDetY,nProj
bound the input geometry; absent variables retain original values.
HECBENCH_VOLUME_OUTPUT dumps the complete double volume. The isolated native
adapter adds only these same size overrides and output dump; algorithm and
native sibling remain unchanged. Random input initialization occurs before
IRIS initialization to preserve srand(123). Numeric comparison uses every voxel.
