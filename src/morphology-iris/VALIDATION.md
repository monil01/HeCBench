Validation on AMD HIP

Status: ok. Native HIP and actual IRIS HIP backend3 pass both32x32/64x64 kernel cases at1024x1024 images,original100 repeats. Complete1048576-byte images match exactly in both cases; sum255 and exactly one white pixel at(611,611), preserving original per-repeat spatial shift. Default8192x8192 images and100 repeats retained. Full default image sizes not revalidated this attempt. Native sibling untouched; output dumps only in copied native instrumentation.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/morphology-iris/attempts/attempt_05` (local, not committed).
