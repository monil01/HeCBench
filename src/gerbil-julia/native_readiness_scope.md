# Gerbil native counting readiness

The active source genome-counting phase uses concurrent addKernel, copies the complete table, and handles overflow through vendor sorting/CPU extraction. Its optional compressEntries call is explicitly commented out in KmerCountingHashTable::extract. Therefore a dormant compaction helper cannot be substituted for the active full CLI pipeline or automatically counted as active work.

Initial source-unchanged GPU readiness probe: valid canonical 28-mer keys in original UInt64 KMer layout (two UInt32 words), 257 input keys with duplicates, 1,024 table entries, source32threads/32keys-per-block, original30probing trials. Complete independently derived per-key counts, no overflow, actual GPU input/full table/guards checked. This probe is not five-trial accepted port/performance evidence. If it fails, preserve actual source output and diagnose before target translations; absence of explicit warp synchronization alone is a source concern, not a proven defect or framework limitation.

Full file parsing, minimizer/temp-file partitioning, CPU/GPU scheduling, and biological data scale remain unverified at this readiness stage. No algorithm substitute or generic atomic-counter success is allowed. Optional compaction's shared UInt64 pointer alignment is separately source-observed, unexecuted, and not a terminal failure claim.
