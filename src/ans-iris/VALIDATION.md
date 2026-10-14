Validation on AMD HIP

Status: deferred. Native decoder has an unwritten output byte at lambda1.38 in the 1MB sweep: sentinel-filled copied native HIP and IRIS both return decoded_first=165 instead of expected_first=0 while all other lambdas match. Original native run passes only with unspecified output initialization; native kernel repair is outside authorized scope. Pinned MI100 backend=3, separate prefix allocations and hipcc host excluded device selection, registration aliasing and host compiler differences. Native sibling untouched. No correctness pass claimed; source defaults remain 10MB.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/ans-iris/attempts/attempt_09` (local, not committed).
