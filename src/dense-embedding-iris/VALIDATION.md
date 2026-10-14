Validation on AMD HIP

Status: ok. Native HIP and actual IRIS HIP backend3 each pass all12 dimension/block checks,covering36 kernel outputs per input at atol1e-3,for1024 rows/batch16/default1000 repeats and1089 rows/batch32/repeat2. All original embedding dimensions768/2048/12288,block sizes128/256/512/1024,three kernel variants,RNG123 and full-output CPU comparisons(including zero tails) retained; finite-output checks added. Fixed output DMem iris_rw to upload zero resets. Default100000 rows/batch256/repeat1000 unchanged; full default row/batch size not executed. Native sibling untouched.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/dense-embedding-iris/attempts/attempt_02` (local, not committed).
