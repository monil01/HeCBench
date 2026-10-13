ECL-MIS IRIS DMem port preserves graph format, hash priorities, 24 blocks x256,
repeat argument and internet.egr default. Original property verification retained;
explicit PASS/FAIL and FNV checksum added, and identical summaries added only to
local HIP validation driver. Full status-array checksum compared exactly.
No changes to native HIP/CUDA source. All device intermediates iris_rw; host updates
use DMem.update, results flush before validation. CUDA backend unvalidated.
