Bonds uses IRIS data memory and the original bond pricing device routines.
Defaults remain one million bonds and 100 repetitions in `make run`.
Validation can use `RUN_ARGS=1` with the same full set of bonds. The native HIP
numeric summaries (sum of dirty prices and four middle-bond values) are the
external comparison. The original CPU pricing calculation is a reference only.

Set HECBENCH_DUMP to a file path to save a uint64 bond count followed by all
dirtyPrice, accruedAmountCurrDate, cleanPrice, and bondForwardVal float arrays.
Whole-output validation uses all one million original seed-123 bonds with one
repetition and compares all four million values against native HIP. This
supersedes the earlier aggregate and middle-bond-only evidence.
