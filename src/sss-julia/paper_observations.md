# SSS — full source workflow, qualified host reuse

RQ1: All 20 captures (242,952,380 bytes each) are exact, including actual input prefixes, full GPU predicates/guards, source search states and model lists. Each trial checks 12,469,275 GPU decisions independently by MCS. The additional saved decoder checks 2,288 states, 6,460 distinct chordal graphs and all 5,099 writer fields. This does not validate the Bayesian posterior independently.

RQ2: CUDA.jl required three program evaluations and two host API repairs before one successful device-JIT process; KA/JACC each passed the first program/device candidate. Each owns two device bodies. Actual token windows are retained with cached/reasoning subsets; shared planning, prior JACC design and future-word2vec planning prevent clean exclusive-cost calibration. No historical counts are invented.

RQ3: Mean trials 2–5 (seconds): native 34.5383133; CUDA.jl 34.0182773; KA 34.4115183; JACC 34.7128937. These are complete instrumented search-loop times with the identical original scientific C++/GSL CPU controller, not GPU-kernel-only times or a causal frontend comparison. No time selected for speed.

RQ4: Own cooperative barriers replace implicit warp communication; the final Boolean path result has one writer after OR. CUDA uses variable dynamic workspace, KA fixed local workspace, JACC explicit 8/384-byte dynamic workspace plus actual checked readonly template preloads. Source CPU/GPU fractions, roofline coordinates and causes of small time differences are unmeasured.

RQ5: All three express the two genuine device predicates. The retained CUDA/GSL host C ABI remains a backend/integration obligation and is not a pure Julia host port or crossvendor portability certificate.

Source GPU termination is preserved: the nominal two-mode bound resets k=0 then breaks after recording one mode. All supplied 150×50 data/init1 search work is exercised; alternative CPU restart branches and inference/application quality are not certified.
