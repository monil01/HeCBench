# Scoped runtime-only applicability — five proposed additions

Private classification-only proposal for native `memcpy` and all four selected `streamCreateCopyDestroy` models. Preserve existing seven canonical applicability rows; add these five only after root independent review and exact current capture/provenance revalidation. No shared file has been changed.

The existing review schema/consumer in `completion_reviews/infra_am/runtime_no_kernel_applicability.json` and `tools/v100_roofline_summarize.py` is reused: exact raw parent provenance SHA, successful rc0/no stop, retained derivation failure reason, source bindings and five correct target rows. Proposed terminal status is `roofline_not_applicable_no_device_kernels`, narrowly qualified by source/runtime capture.

## Source scope

- **Native memcpy:** sixteen original sizes × two Int32 patterns. Each of32cases performs R20 pageable H2D runtime copies plus completion, then R20 D2H copies plus completion; directional means are summed. Setup/reset/full actual readbacks/guards/final free are outside those timer intervals. The included original CLI is renamed and uncalled. Retained source overlay differs only in canonical include and private output filename prefix, with exact inverse proof. No selected kernel launch is authored.
- **streamCreateCopyDestroy all4:** eight runtime cases, two modes across test0/5/10/15,4096Float32 payload words/buffer. The timed pipeline submits pageable H2D copies with stream create/synchronize/destroy, or defaultstream copies/device completion. Setup/allocation/poison/readback/full guards/API-count checks and frees are outside the pipeline timer. Julia storage allocation uses its respective API; setup or runtime internals may use implementation-dependent SM work. No authored device kernel is invoked by the selected primary pipeline.

Each saved NCU process returned0, reported all five target checks true, and explicitly warned `No kernels were profiled.` The CSV has no ID header, so the historical parser failure remains verbatim. This is a reason to adjudicate scalar device-kernel FLOP/DRAM roofline applicability, not to insert zero counts, claim zero HBM/PCIe traffic, prove globally absent SM activity, or label a framework/API unsupported. Runtime/copy-engine/host tracing would be a different measurement method; the saved primary timing/numerical evidence remains separate.

## Evidence limits

The proposal checks compressed/uncompressed frozen manifest hashes, raw provenance/log/CSV/auxiliary-manifest joins, exact current selected CID/config/result joins, bound current sources, five full target check rows, complete artifact byte sizes, and all25 profiling-output SHA identities against accepted primary output SHAs. Where full binaries remain, it rehashes the bytes only; this is not an independent arithmetic replay. Where established scratch cleanup already removed them, the report explicitly records hash-only historical evidence and does not claim new binary recertification. Original failed/no-header CSVs and all source/numerical/timing history remain unchanged.

No source aliases, scientific-family ownership, measured phase, whole-ROI roofline, causal bottleneck, numeric criterion or primary mean is rewritten. Metrics remain null rather than zero. Proposal adoption is solely root's decision.

All analysis is CPU stdlib source/JSON/hash comparison: no GPU/query, target imports, build, tests, Git, sudo or shared scientific mutation. A separate prospective shared review receipt is attached; inherited/cache context and coordination are not exclusive translation cost.
