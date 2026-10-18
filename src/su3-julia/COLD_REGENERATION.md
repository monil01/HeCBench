# Cold reproduction of large CPU fixtures

CPU only; Python3.12.3, NumPy2.5.4, g++13.3.0. Install NumPy with cpu_requirements.txt into an isolated environment, then run `python3 src/su3-julia/reproduce_large_fixtures.py`. The script creates a temporary source tree, compiles the original CPU initializer witness, regenerates both full-size assets and all case descriptors, and requires exact recorded hashes. Measured assets are not overwritten. See cold_generation_proof.json for every byte count/hash.

For a fresh checkout without existing large binaries, compile source_initializer_witness.cpp and run it from .porting-state/portable_julia/su3_reference_review, saving stdout to initializer_layout.json; then run generate_fixtures.py. Compare the generated assets against cases.json/cold_generation_proof.json before any execution. Do not silently update configuration provenance on mismatch.
