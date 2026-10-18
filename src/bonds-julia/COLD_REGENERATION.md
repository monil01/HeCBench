# CPU fixture regeneration

The 204,056,100-byte input and 40,011,000-byte oracle are generated locally, not committed. Their exact SHA-256 values, recipe hashes and successful isolated cold reproduction are in `cold_generation_proof.json` and `cases.json`. Raw GPU captures are never committed.

From the repository root, use Python 3.12.3, NumPy 2.5.4 (`cpu_requirements.txt`), and g++ 13.3.0:

```sh
mkdir -p .porting-state/portable_julia/bonds_reference_review
g++ -O3 -std=c++17 src/bonds-julia/source_cpu_witness.cpp -o .porting-state/portable_julia/bonds_reference_review/cpu_witness
g++ -O3 -std=c++17 src/bonds-julia/generate_source_inputs.cpp -o .porting-state/portable_julia/bonds_reference_review/generate_source_inputs
.porting-state/portable_julia/bonds_reference_review/cpu_witness > .porting-state/portable_julia/bonds_reference_review/ABI.json
(cd .porting-state/portable_julia/bonds_reference_review && ./generate_source_inputs 1000000 256 > source_generation.json)
python3 src/bonds-julia/independent_default_oracle.py
python3 src/bonds-julia/generate_fixtures.py
python3 src/bonds-julia/reproduce_large_fixtures.py
```

The last command independently regenerates in a fresh temporary source tree and compares all five generated assets without overwriting measured bytes. Input generation retains actual srand/rand expressions and unnormalized source day-zero settlements; the oracle derives the source default one-final-payment regime using independent FP64 formulas, not GPU outputs. The first256 actual CPU outputs are a separate witness, not the oracle constructor. Use the four-model study runner after matching pinned Julia/package/toolkit versions and selecting the authorized GPU; its checked five trials retain all output artifacts locally. Absolute study paths in `study_config.json` must be relocated consistently on a different checkout.
