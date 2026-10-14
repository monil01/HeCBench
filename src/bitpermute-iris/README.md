Bit-reversal permutation using IRIS DMem and HIP kernels.

`make run` retains the original 100-repeat workload and all five domains.
HECBENCH_BITPERMUTE_MAX_LOG optionally omits the large log27/log28 domains.
Bounded validation sets it to15 and uses one repetition, testing the original
1024-, 2048- and 32768-element domains against unchanged native HIP kernels
and their exact CPU reference. This exercises direct and tiled permutations;
large-domain numerical correctness is not established by the bounded check.
The native sibling remains untouched; the validation fixture applies only
the same opt-in domain bound to its isolated host copy.
