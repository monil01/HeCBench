# Logan named source-kernel study

Predeclared before GPU execution: seven exact integer sequence-segment cases, 128 threads/block, XDrop100, both source directions in two distinct nonblocking streams. Full endpoint scores checked independently against dense Needleman–Wunsch; separate CPU anti-diagonal source-policy metadata. Target/database is reversed in the kernel coordinates. Exact0 tolerances. Five checked trials, mean2–5.

ROI: two kernel launches + full context completion. Host segmentation, memory allocation/reset and all D2H capture excluded symmetrically. This differs from original CLI timing including D2H results; no whole-CLI performance claim. Scratch stride max segment length+2 rather than original min segment length, explicitly a named allocation-safety mapping. Native device kernel remains byte-unchanged.

Source SeedL constructor/copy constructor does not initialize seedLength; kernel never reads that member. Capture canonicalizes only that dead output member to0 after ROI, with explicit exclusion from byte-equivalence claim; nine meaningful output fields, scores and whole initialized scratch are checked. Actual incoming seed member is deterministic17. No uninitialized member is used to justify numeric acceptance.

No multiGPU, longread, realgenome quality, fullCLI, arbitrary parameters or portable-backend certification. Native warp reduction replaced in target by numerical-equivalent synchronized binary max tree; writes to identical boundary values made single-writer. Those mapping changes prevent intrinsic throughput/warp equivalence conclusions.

Compiler control alignment before framework ports: first native reference series used nvcc default registers (all5 exact oracle/reference pass); subsequent native series restores original Makefile -maxrregcount=32. Saved first series retained, no speed-selected retuning. CUDA.jl device candidate unchanged and its exact captures reused with prefix revalidation.
