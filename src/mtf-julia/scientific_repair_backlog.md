# MTF: full online-transform repair backlog

The original CPU reference demonstrably overflows its 256-byte alphabet on the first nonzero symbol. Native Thrust uses the same zero-filled alphabet and invalid suffix-copy range. Merely initializing an alphabet permutation does not implement standard online move-to-front indices: for ASCII ABA, an independent standard encoder produces [65,66,1], while the remaining source global in-place relabeling produces [66,65,66]. The final-dictionary pass aliases newly written output values with later symbols.

A corrected full online encoder is a separate algorithm study, not an untouched native translation or one-line repair: initialize a defined 256-symbol byte alphabet, emit each current rank before moving its symbol, handle all byte values/signedness, compare an independent decoder round trip and exact per-position reference outputs, and define the same timing boundary in all four implementations. No corrected encoder or GPU timings are claimed here. Keep original source and this diagnostic history intact.

## Corrected secondary study completed

The separately authorized full online encoder now passes all4 models/all5 trials in corrected_secondary_scope.json, result20261018T023729446538Z_mtf. This is an algorithm+memory-safety correction, excluded from untouched-native aggregates. Original source/ASan/semantic diagnostics above remain preserved and invalid; no retrospective originalprogram PASS. Exactrank/finalalphabet/decoder/guard evidence is in corrected_independent_witness.json.
