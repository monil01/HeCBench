Bonds uses IRIS data memory and the original bond pricing device routines.
Defaults remain one million bonds and 100 repetitions in `make run`.
Validation can use `RUN_ARGS=1` with the same full set of bonds. The native HIP
numeric summaries (sum of dirty prices and four middle-bond values) are the
external comparison. The original CPU pricing calculation is a reference only.
