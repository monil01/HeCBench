IRIS dmem port of the Rodinia B+ tree benchmark.

The benchmark data is expected at `../data/b+tree/mil.txt` and
`../data/b+tree/command.txt`, matching the CUDA and native HIP versions.
Build with the local IRIS checkout using `make`; HIP execution requires
`IRIS_ARCHS=HIP` and an available AMD GPU.
