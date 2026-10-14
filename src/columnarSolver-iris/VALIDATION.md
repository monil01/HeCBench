Validation on AMD HIP

Status: deferred. Original scoring archive MD5/size/tar-content verified. At8192 climbs all406528 candidate integers match native HIP exactly and range0..25, but both original plaintext checks FAIL. Full150000-climb IRIS original plaintext check PASS55.47s in attempt3; native full search timed out120s. Native seed0 float32 RNG trace proves direct key[30] access at climb118109/draw1114661/state2147483645; causal link to timeout is not established. Native algorithm/parser repair outside authorized scope. Attempt3 native dump-after-free instrumentation mistake preserved; corrected attempt4 pre-free dump valid. No full native correctness pass, so deferred despite IRIS full PASS. Defaults unchanged.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/columnarSolver-iris/attempts/attempt_04` (local, not committed).
