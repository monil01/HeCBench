# blockScan: exact exclusive integer prefix scans

## Verified scope and algorithms

The study evaluates1024-item Int32 exclusive-prefix scans and the total aggregate on one block, all six native thread/item geometries, three native CUB strategies and three varied fixtures (54 paths). Every output and eight exact guards on each side are retained. Native kernels are unchanged. Historical CUDA.jl implements a genuine serial thread0 scan and is preserved; its printed CUB strategy labels do not represent distinct Julia algorithms. KA and JACC each implement one cooperative Blelloch scan, compared against all three CUB strategies. These are explicit algorithm/mapping comparisons, not intrinsic framework performance measurements.

Default multi-block CLI runs are not certified: all native blocks write coincident output addresses. The study uses grid1 without changing the native kernel, and does not infer an observed wrong answer from the static source hazard. Arbitrary lengths, tails and other precisions are outside the contract.

## Numerical and timing protocol

`fixture_description.json` declares exact tolerances before measurement. All3072 Int32 input values are serialized as an exact artifact prefix. Every path records1024 exclusive prefixes, the aggregate and16 exact guards. Independent Int64 CPU oracles, the adjacent-prefix difference identity and a sum-of-absolute-values bound establish exact integer arithmetic for the fixtures. `generate_fixture.py` reproduces the binary and checks existing bytes. No floating tolerance or aggregate-only acceptance is used.

All18 native algorithm/geometries are checked on each of three fixtures. Portable algorithm labels intentionally repeat the single implemented scan strategy, rather than claiming three independent implementations. Timings cover the complete54-path launch suite, divided by three internal repeats. Allocation, resets, H2D, CPU oracle and readback/artifacts stay outside. Explicit compilation warmup precedes five trials; only the first time is excluded, and trials2–5 are averaged. All raw samples remain available.

KA requests4KB local memory and22 workgroup barriers. JACC requests4096 bytes explicitly and uses the installed `JACC.shared` full-vector template initialization, followed by the same scan barriers. Source-requested operations/resources are distinct from profiled hardware traffic; no bottleneck or occupancy mechanism is claimed without counters.

## Reproducibility and source history

Original CUDA.jl source and Makefile are byte-preserved under `reference_history/`; the nested CUDA.jl source only adds an autorun guard around the historical entrypoint. Native CUB attribution/license text remains in the original included source. Nested KA/JACC main files and current configurations are explicit execution dependencies. `independent_witness.json` contains integer bounds and fixture sensitivity. Actual candidate/check feedback, per-model ontology and prospective usage windows are retained in local study state. Unknown historical costs remain unknown; processed token windows include cached context and coordination, and overlapping phases cannot be summed.

Root alone commits shared files and each verified variant separately. Failed JACC candidate archives belong to JACC history, not the shared/native reference commit.
