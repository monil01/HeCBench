# GREP: complete GPU regex compilation and matching

Primary: ten selected source regex features, three corpora (one line, the full 4,770-line recovered Romeo and Juliet text, and 81,923 deterministic grid-stride controls). Source launch geometry 512 blocks ×160 threads is preserved. Every timed kernel performs simplified-infix parsing, postfix conversion, Thompson NFA construction and full-string byte matching on the GPU. There is no CPU-precompiled-NFA substitution.

## Reproduce

From repository root, Python3.12.3 standard library and pinned Julia/CUDA environments:

```sh
python3 src/grep-julia/fetch_dataset.py
python3 src/grep-julia/generate_assets.py
python3 src/grep-julia/generate_graph_oracle.py
python3 src/grep-julia/prepare_cold_config.py /tmp/grep_cold.json
python3 tools/verify_portable_julia.py grep --config /tmp/grep_cold.json
```

The fetcher verifies declared DVC archive size630879B and MD5. The generators exactly reproduce CPU input/output/graph fixtures and parameters; the committed reviewed protocol is preserved. Raw GPU captures stay local. `audit_saved_outputs.py` checks the original local full bundle and isolated CPU regeneration without GPU work.

## Evidence and boundaries

All five trials check full raw Int32-encoded actual inputs, output Boolean masks and serialized guards against Python re.fullmatch. Only timing trial1 is excluded. Supplemental observers after all five timings capture canonical postfix/NFA/state edges for every one of 512 blocks per pattern; these are not timed internal captures. The graph control is source-semantic; the independent matching oracle is Python's separate regex engine.

Native CUDA bodies are unchanged. Julia maps pointer patch lists to Int32 indices, packs postfix bytes in shared words and uses a flat private workspace. These change storage/resource behavior. CUDA.jl's host-parse rejection and GPU heap-allocation failure are archived; no heap limit or tolerance was relaxed. KA/JACC inherit the corrected shared helper. Only CUDA backends are tested.

Original full Romeo CLI checks pass all ten patterns. A separate original CLI witness shows the last record is lost without a terminating newline; primary terminated inputs avoid that boundary. Regex/state/list capacity limits and absent epsilon-cycle protection are preserved, not silently repaired or called framework limitations. Read `graph_capacity_proof.json`, `native_readiness.json`, `study_adjudications.json`, and `paper_observations.md`.
