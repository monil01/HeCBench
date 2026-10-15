Validation on AMD HIP

Status: verified. Terminal-symbol repair is documented in README.md. Native HIP and IRIS pass every original-input byte check for all15 lambda sweeps at 1MB and original10MB default with sentinel165 output prefills; complete decoded streams are byte-identical. The earlier unwritten-byte failure is preserved in attempts01..09 and untouched native sources in attempt10. Source defaults remain10MB.

Run with IRIS_ARCHS=HIP from this directory. Reproduce using `HECBENCH_VALIDATE_SENTINEL=1 IRIS_ARCHS=HIP ./main 10`. Attempt10 contains native/IRIS build/run logs, full-output artifacts and comparison evidence locally under `.porting-state/porting_logs/ans-iris/attempts/attempt_10` (not committed).
