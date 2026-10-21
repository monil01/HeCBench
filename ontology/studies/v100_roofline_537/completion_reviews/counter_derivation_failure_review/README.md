# Seven latest derivation failures: applicability review, not recapture backlog

Seven latest raw statuses remain `profile_completed_counter_derivation_failed`:

| Source | Models | Actual process | Counter report | Selected effective status |
|---|---|---|---|---|
| mallocFree | CUDA, CUDAjl, KA, JACC | rc0, no guard stop, full5checked trials | log-only335–383B, explicit no kernels profiled | roofline_not_applicable_no_device_kernels |
| memcpy | CUDAjl, KA, JACC | rc0, no guard stop, full5checked trials | log-only461B, explicit no kernels profiled | roofline_not_applicable_no_device_kernels |

Exact parser cause in all seven: `No NCU CSV ID header`. There is no numeric metric table. This is not a unit conversion/missing metric/CSVfield-limit failure. Process connection warnings about helper processes terminating before instrumented API calls coexist with a successful complete primary target and full checks; they are not evidence of numerical failure.

Existing infra_am/runtime_no_kernel_applicability.json SHA-bound each exact raw parent provenance and source, and root summarizer already maps its effective selected status to source-NA only when parent SHA and successful rc0/no-stop gates match. This review independently rechecked those joins, full5trial rows and source hashes under metadataLOCK_SH. Latest raw failures remain unchanged/history; the canonical selected effective status is deliberately different.

mallocFree source uses the runtime bridge allocation/free lifecycle (host-runtime API operations, no authored device algorithm), with allocation/non-null/free status evidence; no numerical output. Julia versions explicitly retain runtime interoperability instead of replacing allocation pools. Memcpy selected Julia paths use copyto!/KA.copyto!/JACC.transfer! for host↔device pageable buffers; full exact data+guards check retained. Source APIs and NCU no-profiled-kernel observation do not prove no driver/internal SM activity; do not infer zero traffic, zero FLOPs or zero hidden runtime work.

Actions: retain existing classification; zero new GPU jobs, zero NCU2025 candidate necessity for these unchanged contracts. No duplicate repair is required. Same-model alternative raw plans are enumerated in reviewed_actions.json; empty lists mean none found in actual rawbatch tree at review. Any existing collector encountering terminal raw provenance preserves it; generic039/041 considers selected operational profile failures only, so these seven effective source-NA outcomes are outside that frontier. The separate native memcpy blocked plan is NOT one of the seven.

Heartwall's saved-counter derivation recovery is inappropriate: it requires real metric rows, duration/DRAM and saved device identity. No parser patch can recover absent hardware data. A different scientific question—allocator latency, runtime API timeline, pageable PCIe transfer bandwidth—needs a separately named protocol and is not a repair that fabricates kernel roofline points.

No changes to collector32a9, classifier,003 calibration, source, raw files or sharedmetadata. No GPU query/run/build/test/target imports. Full snapshot, raw/frozen hashes, selected joins and source excerpts are retained privately. Observed usage shared review only; primarytimings unchanged.
