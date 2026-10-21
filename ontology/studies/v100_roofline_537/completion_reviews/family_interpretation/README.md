# Twelve-family scientific interpretation

Read **INTERPRETATION.md** for the source-backed explanation and compact
paper-ready paragraph. Final machine-readable evidence is **reviewed_evidence/**:
operation/traffic CSV, exact source arithmetic models, source excerpts, raw
counter SHA bindings, compiler/profiler receipts and selected-SASS availability.
`evidence/` is the preserved first metadata-analysis revision; no extra GPU runs.

## Principal findings

1. Jacobi's502,333,920 common stencil-plus-pointwise-square FLOPs plus
   356,515,840 native shuffle/block FADD equal858,849,760 exactly. The source
   operation census matches every captured instruction-class count. Floating
   atomics are excluded from this counter set, so Julia's atomic reduction work
   is omitted: do not treat the values as complete total-FP-work equivalence.
2. Native Triad and dot use FFMA where the Julia captures use separate FADD/FMUL;
   weighted FLOP totals remain equal. No exact rounding/bitwise claim is inferred.
3. JACC dot/Jacobi have higher measured DRAM writes, not more counted arithmetic.
   The shared-template copy is not a declared global write, and per-point atomics
   are also used by KA. These source facts do not determine compiler spill,
   atomic lowering, or cache/replay behavior. Those remain hypotheses.

All source/counter/strict-audit inputs are hashed and rechecked. No selected
SASS/PTX/cubin was retained in the capture manifests; cleaned scratch binaries
were not recreated. No register/spill claim, causal bottleneck, whole-host-ROI
rate or primary-timing replacement is made. `core_work_diagnostic.json` is an
explicitly auxiliary common-work normalization, not a modification of the frozen
executed-counter points or a complete norm-reduction model.

## Reproduce existing-data analysis

```sh
python3 analyze_family_evidence.py \
  --source-root /home/imo/HeCBench_julia \
  --output-root /tmp/FRESH_FAMILY_INTERPRETATION
```

The script reads actual local source/counter paths from frozen receipts; it does
not compile/JIT/build, run tests, execute GPU work or change sources/shared data.
The NVIDIA reference provides only a general possible mechanism, not proof about
these captures. The observed usage receipt is shared analysis with marker lag,
not exclusive billed cost or historical translation accounting. Root owns Git,
paper/ontology integration and any subsequent profiling decisions.
