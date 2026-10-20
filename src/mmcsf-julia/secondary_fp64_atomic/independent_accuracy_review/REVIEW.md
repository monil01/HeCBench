# MMCSF: bounded review of global-accumulator precision intervention

**Assessment:** the proposal is source-plausible as a **qualified mixed-precision secondary**, not as proof that native Float32 arithmetic was defective, not unchanged-native agreement, and not evidence that the secondary already passes. Preserve the original strict rejection and source0.1 counterfactual. This review performed no GPU execution, target drafting, source/status/Git edits or package changes. Only the separately requested usage meter writes its boundary events.

## Source findings

`source_review.json` binds inspected files by SHA256 and lists all output uses. Seven GPU entrypoints have nine atomic-add sites. Every `dU0` use in those kernels is an atomic destination; none loads the output for subsequent computation. Tensor/factor inputs and private `tmp`, `tmp_val`, `tmp2` variables are `DTYPE` Float32. Therefore changing **only output pointer types and global atomic accumulation** can leave private fiber/product arithmetic unchanged in source. Do not change the global DTYPE definition.

All factor modes currently share a single packed Float32 `dU` allocation. The host permutes its views according to each tile's `modeOrder`. For one MTTKRP mode, every dispatched partition must accumulate into the **same** separate DoubleOut, even though seven different branch signatures may be used. Assert that the original packet slot8 points to the current target-mode slice; preserve slots9…11 as the original Float32 input-factor views and prove they are disjoint from DoubleOut. Do not accidentally allocate/reset DoubleOut per partition or dispatch.

Source host code restores the preceding output mode to constant `mode+0.5` before advancing to the next mode. These outputs are **not** fed back as ALS factor updates in the current source driver. Thus the critical dependency is accumulation across all partitions **within a mode**. Capture and verify the projected old Float32 `dU` before the source advances/reset occurs. Retain the full before/after packed-factor snapshots: non-target modes must remain bit-identical.

## Minimal patch surfaces, without implementing them

1. Seven kernel output parameters: `DTYPE* dU0` → explicitly typed Double output pointer; all input and private arithmetic remains Float32.
2. Nine atomic sites: convert the **completed Float32 contribution** to Double. In particular op4's `tmp*dU2[...] * dU3[...]` must finish its original left-associated Float32 products **before** conversion. `double(tmp)*...` would change private products and violate the proposed boundary. Explicit conversion at the call boundary makes the policy auditable.
3. Native dispatcher emitted by `prepare_interceptors.py`: its slot8 cast is currently `float*`; update only the output cast for the qualified variant.
4. `MMLaunch` is a void-pointer ABI and can retain its256-byte layout, but packet/span logic currently assumes slot8 element width4. Retarget to DoubleOut with its actual Float64 width/capacity. An approach is constructing the original packet for its Float32 source views, validating the original output slice, then creating a qualified packet with slot8/length8 replaced. Do not reinterpret the old Float32 allocation as Double storage or change `mm_make3/4` input-factor signatures globally.
5. Allocation/canonical bookkeeping: `mm_malloc` currently enumerates original source allocations and assigns `dU` kind4. New DoubleOut needs explicit separately tracked storage/guard provenance. Do not silently insert it into that original sequence, alter old canonical assets, or truncate/overstate its callback view.
6. Mode lifecycle: allocate/reset the same guarded Float64 target, execute all branch dispatches, complete them, retrieve DoubleOut, project round-to-nearest into a Float32 host buffer, and copy that buffer into the original target `dU` slice. Then perform original source checks and added independent checks.

## Timing trap

`Observe` currently subtracts the entire lifetimes of `mm_mode_begin` and `mm_mode_end` from the overall timer. New algorithmic allocation/zeroing and **D2H DoubleOut → CPU projection → H2D Float32 output** placed inside those scopes would be silently excluded. Move/split those boundaries explicitly: all new initialization, projection, necessary transfers and completion belong inside the declared identical all-four ROI. Only added readbacks solely for evidence, guard checking, COO oracle construction and file writes may be excluded as declared observations. Avoid charging a shared transfer twice or subtracting a transfer required for projection merely because its retrieved values are also captured. Record buffer allocation/reuse and host temporary-allocation policy consistently.

The existing last-mode copyback/source checker remains distinct from the new all-mode strict independent acceptance. Source0.1 acceptance cannot override unchanged strict0.01+2e-5|gold| rejection.

## Quantitative support and its limits

`row_precision_bound.json` reproduces the proposed **CPU model** for row762/mode0 from the retained canonical CSF arrays. It uses actual Float32 tensor values, IEEE Float32 `fmaf` for private accumulation, original left-associated Float32 products, and exact rational sums/error arithmetic. It does not capture the actual GPU contribution trace or atomic linearization.

For68,715 nonzeros and67,645 modeled rounded contributions:

| Component | Absolute error/bound |
|---|---:|
| Exact COO sum |409.51111343252705|
| Exact sum of modeled Float32 contributions |409.5112610885408|
| Private-product/fiber error |1.4765601372e-4|
| Any-order Float64 global-add bound |5.3520856466e-6|
| Conservative Float32 projection bound |2.4408773568e-5|
| Combined projected error bound |1.7741687294e-4|
| Unchanged strict budget |1.8190222269e-2|

For actual rounded contributions c_i, finite normal Float64 summation in any order satisfies an absolute bound γ_(m−1)(u64) Σ|c_i|, where γ_n=nu/(1−nu). Converting each finite Float32 contribution to Float64 is exact. Total error versus exact COO is bounded by private-contribution error plus this global-add term. A conservative round-to-nearest Float32 projection allowance is u32(|Σc_i|+global_bound)+2^-150. The report evaluates this particular modeled row using rational arithmetic. It is **not** an all-row guarantee or causal attribution of the original GPU error. Keep CPU-model/FMA assumptions explicit and verify actual GPU outputs independently.

For all-row validation, either obtain/cross-check each source branch's rounded contribution model and private-error budget, or simply enforce the existing exact COO scientific gate on **every actual Double output and projected Float32 output** for all modes/cases/trials. Double promotion does not remove private Float32 fiber errors; the one successful modeled row cannot justify accepting another failed field.

## Controls, qualifiers and unmeasured cases

- Declare the secondary precision policy prospectively: Float32 tensor/factors/private arithmetic; Float64 global accumulator; Float32 CPU projection for source compatibility. Keep tolerance unchanged. This is an accuracy intervention/arithmetic-policy change, distinct from any already justified host preprocessing repairs.
- Preserve all original primary failures and small controls; use a new configuration identity and fresh symmetric native/CUDA.jl/KA/JACC series. Retain all actual precision-specific buffers, guard words and per-mode initialization inputs.
- Compare actual DoubleOut and projected Float32 buffers independently with exact COO, and verify the projection's own expected rounding bit pattern. Record finite/overflow status; do not canonicalize NaNs or relax a failing field.
- Preserve Float32 FMA/contraction, operation order, products and any relevant FTZ policy. Float32 global atomic behavior for subnormals can differ from Double atomic accumulation; audit whether actual inputs/contributions can reach that regime rather than asserting universal equivalence.
- Expect possible tiny Float64 atomic-order differences; higher precision does not establish bitwise repeatability. Use declared numerical tolerances and full retained values, not forced scheduling or output canonicalization.
- Seven branch bodies contain scalar private temporaries around rank loops; ranks>32 can mix columns in some original expressions. Current prepared ranks≤32 avoid this issue. Do not generalize the result to arbitrary rank or silently repair those source expressions while claiming output-pointer-only promotion.
- Existing factor columns are repeated constants. A separately declared small nonconstant-column control with rank≤32 would strengthen indexing/factor validation without changing the original-size primary. This is a recommendation, not a requirement inferred after a failed result or a performed experiment.
- Mixed-precision atomic traffic, added DoubleOut allocation/initialization, CPU projection/transfers and host orchestration change the runtime problem. Exclude from unchanged-native/unchanged-precision aggregates and avoid causal frontend/roofline claims without controls/counters.
- Source outcomes remain distinct: original strict accuracy failure, scientific precision intervention, and target/framework compiler or numerical failures. No observed framework limitation follows from the original Float32 rejection.
