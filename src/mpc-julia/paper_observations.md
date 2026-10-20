# MPC: lossless scientific-data bit codec

RQ1: all40 captures/708641280bytes exact in native CUDA, CUDA.jl, KA and JACC; independent full-format encoder and decoder plus direct actual-input/decoded equality. Full290MB native source trace roundtrips separately. Native n=1 host allocation3words vs30written is observed by sanitizer; conservative host-capacity-only secondary passes sanitizer and bit roundtrip. No GPU arithmetic repair.

RQ2: one source bundle and two declared configuration/JIT process checks per target; no GPU/compiler/numerical repairs. Shared two-kernel/nine-helper math created in CUDA.jl, reused by KA/JACC. Shared native preparation and verification costs are separate. KA/JACC windows have disclosed forward-planning/left-censoring and are excluded from per-target cost calibration.

RQ3: aggregate primary33case means native2.054655ms, CUDA.jl2.311839ms, KA2.411620ms, JACC5.623437ms. Secondary18case means .596894/.863954/.934493/2.593867ms. Scope includes one1M-real-word dim1 case plus32small dimension controls; not full-trace throughput. NativeNVCC12.0 versus Juliaartifact12.9.41 is an explicit compiler covariate.

RQ4: all target32registers and driver resource calculation permits2CTA/SM. Compiled local allocations32/40B CUDA.jl,32/48B KA,240/224B JACC; JACC dynamic template shared/preload differs. These are associations, not spill-traffic/causal-cost or achieved-occupancy measurements. No floating roofline interpretation for this integer bit codec.

RQ5: this faithful global-ring mapping uses explicit volatile-global PTX leaves and source launch bounds; KA/JACC own-framework kernels integrate CUDA backend helpers/adapters. Generic alternatives/backends are unmeasured, not proved unsupported.
