# Publication figure handoff

Final files are under `final/`:

- `coverage_assessment.pdf`, `.svg`, `.png`: common-denominator status bars and
  scientific-family/name coverage with explicit unmeasured counts.
- `paired_family_intensity.pdf`, `.svg`, `.png`: twelve strict family captures,
  plotted separately by model; no causal or speedup claims.
- Four `plotted_*.csv` files preserve every category, family denominator and
  exact intensity numerator/denominator, including all 2148 name/model rows.
- `descriptor_input.json`, `strict_family_audit_input.json`,
  `plot_source_snapshot.py`, `provenance.json` and `manifest.json` preserve
  immutable inputs, source and all input/output SHA256 values.

The figure source is `tools/plot_v100_benchmark_assessment.py`. It used the
existing `/home/imo/noback/conda/bin/python3.9` environment (NumPy 1.22.3,
Matplotlib 3.5.1); no packages were installed. All source/current ledgers and
benchmark sources remain unchanged. Both PNG previews were visually inspected;
labels and legends are readable, with the five hardware cases per model stated.
Earlier layout versions are preserved in `initial/` and `reviewed/`; these are
figure/documentation revisions, not extra GPU measurements or benchmark attempts.

## Reproduce from the frozen bundle

```sh
/path/to/existing/python3.9 final/plot_source_snapshot.py \
  --descriptor final/descriptor_input.json \
  --family-audit final/strict_family_audit_input.json \
  --output-root /tmp/NEW_FIGURE_DIRECTORY \
  --snapshot-label '21 Oct 2026; partial campaign'
```

A normal source checkout can instead run the tool with only `--output-root`;
its defaults read the frozen study descriptor and latest strict family audit.
Output must be a fresh directory below `/tmp`. Input hashes are checked before
and after plotting. Unknown/new classifications go to an explicit unresolved
category rather than becoming zeros or silently qualifying for FP positions.
The dated descriptor intentionally reflects a partial campaign; it must not be
presented as a later completion census. Fresh rebuild metadata timestamps may
change binary PDF/SVG hashes; exact plotted scientific values remain in CSVs.

`CAPTIONS.md` offers manuscript-ready scope text. `usage_receipt.json` is observed
shared analysis, including context/cache and marker lag, not translation cost.
No GPU, builds, tests, sudo, Git, paper or shared metadata edits were performed.
