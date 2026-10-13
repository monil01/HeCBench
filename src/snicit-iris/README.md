SNICIT retains the original nine dense/sparse/clustering/recovery kernels,
original A/B/C/D CLI choices, A default, n=10000,b=10000,seed_size128,
original thresholds, densities and actual MNIST/network inputs. No vendor
solver or classifier is invented. No RNG is used by this original algorithm.
Original -n<10000 reads10000 records into smaller arrays; that source bug is
not silently fixed. Bounded tests keep n10000 and explicitly use b1000.

DMem-only host: originally managed host reads flush first, host compaction,
sentinel writes and resets call update; partial/in-place writes use rw.
The original output subrange device-copy stages through DMem-flushed host
backing and update, preserving exactly copied data. Original uninitialized
scratch allocation is retained; no zero default invented. bool remains1byte,
int/float4bytes. P1 static512-float shared capacity covers all fixed supported
network widths128/256 and seed128, with original indexing/arithmetic.

Native adapter changes only original CLI11 include path and opt-in raw-score
diagnostic. Originals untouched. Validate full output scores with atol.001,
rtol.0001 and compare all predicted labels/correct-counts exactly; native
repeat variability is inspected, not assumed away. Numerical evidence in
owned state. Mode D original CIFAR inputs are absent and are not fabricated.
