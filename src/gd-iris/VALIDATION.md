Validation on AMD HIP

Status: ok. Native HIP and actual IRIS HIP backend3 pass the original CPU objective/error reference after default100 iterations, lambda0.0001/alpha10, on deterministic sparse inputs257x512 and513x1024 (8225/16417 nonzeros). Native/IRIS objective and training-error summaries match:0.568403/0.151751 and0.624744/0.302144 within atol1e-4/rtol1e-3; CPU objective max discrepancy1.192093e-7. Full feature blocks and partial observation blocks covered. Original gisette default filename retained; available4x3 fixture remains a known inherited partial-feature hipCUB reduction failure, not repaired. Real gisette dataset/full default input not validated.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/gd-iris/attempts/attempt_02` (local, not committed).
