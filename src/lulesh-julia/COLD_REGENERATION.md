# CPU-generated assets

The22 raw CPU files (actual source meshes/regions/parameters and full serial state witness),151,459,788bytes total, are generated locally and excluded Git. CPU serial witness shares original mathematical bodies; independent physics evidence is `CPU_physical_invariant_audit.json`/`output_audit.json`. Per-file hashes, sizes, compiler/policy and successful fresh-tree proof are in `cold_generation_proof.json`. GPU outputs are never committed.

With g++13.3.0, source CUDA12.0 headers, Python3.12.3 and pinned `cpu_requirements.txt`, from repository root:

```sh
g++ -O3 -std=c++17 -D__host__= -D__device__= -I/usr/local/cuda-12.0/include -x c++ src/lulesh-julia/generate_source_inputs.cpp src/lulesh-julia/source_volume.cpp src/lulesh-cuda/lulesh-init.cu -o /tmp/lulesh-generate
(cd src/lulesh-julia && /tmp/lulesh-generate)
g++ -O3 -ffp-contract=off -std=c++17 -DCPU_REFERENCE -D__host__= -D__device__= -I/usr/local/cuda-12.0/include -x c++ src/lulesh-julia/study_native.cu src/lulesh-cuda/lulesh-init.cu -o /tmp/lulesh-cpu
(cd src/lulesh-julia && /tmp/lulesh-cpu)
python3 src/lulesh-julia/reproduce_large_fixtures.py
```

The final command regenerates in a separate temporary source tree, compares all22 SHA-256 values and leaves measured files untouched. Use pinned Julia1.12.6/CUDA6.2.1/StaticArrays1.9.18/KA0.9.45/JACC1.4.0 and the four-model runner for GPU verification. Relocate absolute study paths consistently for a different checkout; keep raw captures local. Warmup and scientific checks remain outside the declared mixed host/device cycle ROI.
