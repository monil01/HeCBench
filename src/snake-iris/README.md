# Snake IRIS

Preserves four arguments, packed DNA encoding, 256-thread blocks, all26 thresholds
(loopPar0..25), repeat count, original sneaky_snake algorithm and per-read CPU
reference memcmp. Default run100-base reads,30000 pairs,1000 repeats and original
Datasets filename retained. Inputs use DMem iris_r; completely overwritten
results use iris_w and flush after each threshold. No native allocation/launch.
SNAKE_VALIDATION_DUMP optionally writes all per-read decisions for each threshold
in order for exact native HIP comparison; it does not change computation.

