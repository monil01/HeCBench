# Document/profile scoring named study

Original Altera-source compute/reduction kernels run unchanged. Existing genuine CUDA.jl kernels are preserved byte-for-byte in their nested child. Named input:64 ragged documents,160 padded128-word lines split between two DIMM arrays,16384 profile entries and16384-word Bloom filter. Full source default2^24 profile/262144documents is unmeasured; ~410KiB selected data fitsL2, so no DRAM/application scaling claim.

Packed terms carry24-bit IDs and8-bit frequencies, with0xffffffff padding. Actual source mulfp multiplies low20 bits and16 bits starting atbit24, omittingbits20..23; selected weights vary both supported parts and avoid omitted positions/productoverflow. This is not a general UInt64 multiplication certificate. Independent CPU uses ordinary fullUInt64 weighted sums without Bloom filtering; exact64 scores plus160 high/160 low partials checked in every model/trial. Artifacts losslessly widen bounded unsigned values toInt64 outside timer.

Native8threads/line computes8terms perlane perDIMM, then one64-thread documentreduction block. Nativeulong4 finalsum becomes serial8sum in existingJulia; nativeperlane8elementtemporaryarrays become scalarloops. KA retains mapping with64Blocal scratch; JACC addszero-template staging/barrier and64Bshared, reductionshmem0. Both genuineports compile/passfirstcandidate. Five warm inprocess trials,mean2–5,10twokernel internalrepeats. Allallocations/transfers/reset/oracles outside clock.

No framework feature unsupported. Exact billed tokens unknown; finalsource character/4 proxies partial andexcludecontext/failedtext/infrastructure/paper. Scientific role is informationretrieval document/profile weighted termscoring, with noidentified corpus or domain-science application.
