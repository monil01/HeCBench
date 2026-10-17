# mixbench verified configuration

mixbench_original8M_fp32_ci16_repeat20_reset_trials_v1

Only computeiterations16 measured, no arithmeticintensitysweep claim.

NativeFMA contraction vsJuliauncontracted withinoriginal1e-6; no measuredroofline withoutprofiling.

Repeatedidenticalnative stores optimizedbycompiler possible; useful/source/issuedwork distinctions required.

Originalread-onlymixbench retains sparseoutputwrites:32768writtenfloats of8M, remainingzeros fullychecked. Eachtrialresetoutsideclock; kernelstateevolves20launches, originalwarmupstateexcluded.

Five in-process trials, all outputs checked against both references and the configured CPU oracle. First trial time excluded; arithmetic mean of last four. Raw samples/configuration/provenance remain linked in per-variant verification.json. Partial token estimates are not total usage.
